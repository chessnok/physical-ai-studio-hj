from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from db.schema import DatasetDB
from exceptions import CameraDeviceConflictError, ResourceInUseError, ResourceNotFoundError, ResourceType
from repositories.project_camera_repo import ProjectCameraRepository
from repositories.project_environment_repo import ProjectEnvironmentRepository
from robots.catalog.registry import RobotCatalogRegistry
from schemas.environment import Environment, EnvironmentWithRelations
from utils.camera_factory import shared_device_conflict


class EnvironmentService:
    def __init__(self, session: AsyncSession, catalog_registry: RobotCatalogRegistry) -> None:
        self.session = session
        self.catalog_registry = catalog_registry

    def _repo(self, project_id: UUID) -> ProjectEnvironmentRepository:
        return ProjectEnvironmentRepository(self.session, project_id, self.catalog_registry)

    async def get_environment_list(self, project_id: UUID) -> list[Environment]:
        return await self._repo(project_id).get_all()

    async def get_environment_by_id(self, project_id: UUID, environment_id: UUID) -> EnvironmentWithRelations:
        environment = await self._repo(project_id).get_by_id_with_relations(environment_id)

        if environment is None:
            raise ResourceNotFoundError(ResourceType.ENVIRONMENT, str(environment_id))

        return environment

    async def _check_cameras(self, project_id: UUID, environment: Environment) -> None:
        """Refuse an environment that uses one physical camera twice.

        Raises:
            CameraDeviceConflictError: Two of its cameras are the same device.
        """
        wanted = {camera.camera_id for camera in environment.cameras}
        if len(wanted) < 2:
            return
        cameras = await ProjectCameraRepository(self.session, str(project_id)).get_all()
        if (conflict := shared_device_conflict(camera for camera in cameras if camera.id in wanted)) is not None:
            raise CameraDeviceConflictError(conflict)

    async def create_environment(self, project_id: UUID, environment: Environment) -> Environment:
        await self._check_cameras(project_id, environment)
        return await self._repo(project_id).save(environment)

    async def update_environment(self, project_id: UUID, environment: Environment) -> EnvironmentWithRelations:
        await self._check_cameras(project_id, environment)
        repo = self._repo(project_id)
        existing = await repo.get_by_id(environment.id)
        if existing is None:
            raise ResourceNotFoundError(ResourceType.ENVIRONMENT, str(environment.id))

        await repo.update(existing, environment.model_dump(exclude={"id", "created_at", "updated_at"}))

        updated = await repo.get_by_id_with_relations(environment.id)
        if updated is None:
            raise ResourceNotFoundError(ResourceType.ENVIRONMENT, str(environment.id))

        return updated

    async def delete_environment(self, project_id: UUID, environment_id: UUID) -> None:
        repo = self._repo(project_id)

        environment = await repo.get_by_id(environment_id)
        if environment is None:
            raise ResourceNotFoundError(ResourceType.ENVIRONMENT, str(environment_id))

        try:
            await repo.delete_by_id(environment_id)
        except IntegrityError as error:
            await self.session.rollback()
            dataset_names = (
                await self.session.scalars(
                    select(DatasetDB.name).where(
                        DatasetDB.project_id == str(project_id), DatasetDB.environment_id == str(environment_id)
                    )
                )
            ).all()
            if dataset_names:
                raise ResourceInUseError(
                    ResourceType.ENVIRONMENT,
                    environment_id,
                    message=(
                        f"Environment '{environment.name}' cannot be deleted because it is used by dataset(s): "
                        f"{', '.join(dataset_names)}. Delete those datasets first."
                    ),
                ) from error
            raise
