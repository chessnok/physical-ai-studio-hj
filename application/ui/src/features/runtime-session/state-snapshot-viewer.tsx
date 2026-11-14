import { useState } from 'react';

import {
    ActionButton,
    AlertDialog,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogContainer,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    Text,
    toast,
} from '@geti-ui/ui';
import { Delete } from '@geti-ui/ui/icons';
import { useQueryClient } from '@tanstack/react-query';

import { $api, fetchClient } from '../../api/client';
import { getApiErrorMessage } from '../../api/errors';
import { SchemaStateSnapshot } from '../../api/openapi-spec';
import { labelOf, snapshotListKey } from './state-snapshot';

import classes from './state-snapshot-viewer.module.css';

const TOAST = { timeout: 10_000 };

/** The URL of one of a snapshot's images, such as `top.color.png`. */
export const snapshotFileUrl = (project_id: string, snapshot_id: string, name: string): string =>
    fetchClient.PATH('/api/projects/{project_id}/state_snapshots/{snapshot_id}/files/{name}', {
        params: { path: { project_id, snapshot_id, name } },
    });

interface SnapshotImage {
    name: string;
    caption: string;
}

/** What a person can look at: every camera's color image, and its colorized depth if it has depth. */
export const snapshotImages = (snapshot: SchemaStateSnapshot): SnapshotImage[] =>
    snapshot.cameras.flatMap((camera) =>
        [
            { suffix: '.color.png', kind: 'color' },
            { suffix: '.depth_preview.png', kind: 'depth' },
        ].flatMap(({ suffix, kind }) => {
            const name = camera.files.find((file) => file.endsWith(suffix));
            return name === undefined ? [] : [{ name, caption: `${camera.name}, ${kind}` }];
        })
    );

const SnapshotPreview = ({
    project_id,
    snapshot,
    time,
    onClose,
}: {
    project_id: string;
    snapshot: SchemaStateSnapshot;
    time: string;
    onClose: () => void;
}) => {
    const images = snapshotImages(snapshot);
    const comment = snapshot.comment ? `"${snapshot.comment}"` : 'no comment';
    const capturedAt = new Date(snapshot.captured_at).toLocaleString();

    return (
        <Dialog>
            <Heading>Snapshot of {time}</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-200'>
                    <Text>
                        {labelOf(snapshot)}, {comment}. Captured {capturedAt}
                        {snapshot.environment_name ? ` in ${snapshot.environment_name}` : ''}.
                    </Text>
                    {images.length === 0 ? (
                        <Text>This snapshot has no images.</Text>
                    ) : (
                        <div className={classes.images}>
                            {images.map(({ name, caption }) => (
                                <figure key={name} className={classes.figure}>
                                    <img src={snapshotFileUrl(project_id, snapshot.snapshot_id, name)} alt={caption} />
                                    <figcaption>{caption}</figcaption>
                                </figure>
                            ))}
                        </div>
                    )}
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={onClose}>
                    Close
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

/** A small image of the snapshot (its first camera's color image); a click shows all of its images. */
export const SnapshotThumbnail = ({
    project_id,
    snapshot,
    time,
}: {
    project_id: string;
    snapshot: SchemaStateSnapshot;
    time: string;
}) => {
    const [isOpen, setIsOpen] = useState(false);
    const first = snapshotImages(snapshot)[0];

    if (first === undefined) {
        return null;
    }
    return (
        <>
            <button
                type='button'
                className={classes.thumbnail}
                onClick={() => setIsOpen(true)}
                aria-label={`Show snapshot of ${time}`}
            >
                <img src={snapshotFileUrl(project_id, snapshot.snapshot_id, first.name)} alt='' loading='lazy' />
            </button>
            <DialogContainer type='fullscreen' onDismiss={() => setIsOpen(false)}>
                {isOpen && (
                    <SnapshotPreview
                        project_id={project_id}
                        snapshot={snapshot}
                        time={time}
                        onClose={() => setIsOpen(false)}
                    />
                )}
            </DialogContainer>
        </>
    );
};

/**
 * Moves a snapshot to the trash after a confirmation, in a modal dialog: Enter and Escape there
 * act on the dialog, never on the page's episode hotkeys. Nothing is deleted on the backend.
 */
export const TrashSnapshot = ({
    project_id,
    snapshot,
    time,
}: {
    project_id: string;
    snapshot: SchemaStateSnapshot;
    time: string;
}) => {
    const queryClient = useQueryClient();
    // Callbacks on the mutation itself: the row is gone once the list is fetched again.
    const { mutate } = $api.useMutation('delete', '/api/projects/{project_id}/state_snapshots/{snapshot_id}', {
        meta: { invalidates: [snapshotListKey(project_id)] },
        onSuccess: (trashed) => {
            toast.positive(`Snapshot of ${time} moved to trash: ${trashed.moved_to}`, TOAST);
            if (trashed.csv_warning) {
                toast.info(trashed.csv_warning, TOAST);
            }
        },
        onError: (error) => {
            const reason = getApiErrorMessage(error) ?? 'the backend did not answer.';
            toast.negative(`Could not move the snapshot of ${time} to trash: ${reason}`, TOAST);
            void queryClient.invalidateQueries({ queryKey: snapshotListKey(project_id) });
        },
    });

    return (
        <DialogTrigger type='modal'>
            <ActionButton isQuiet aria-label={`Move snapshot of ${time} to trash`}>
                <Delete />
            </ActionButton>
            <AlertDialog
                title='Move this snapshot to trash?'
                variant='destructive'
                primaryActionLabel='Move to trash'
                cancelLabel='Cancel'
                autoFocusButton='cancel'
                onPrimaryAction={() => mutate({ params: { path: { project_id, snapshot_id: snapshot.snapshot_id } } })}
            >
                The snapshot of {time} ({labelOf(snapshot).toLowerCase()}) leaves the list and snapshots.csv. Its folder
                is moved to state_snapshots_trash in Studio&apos;s storage folder, from where it can be moved back.
            </AlertDialog>
        </DialogTrigger>
    );
};
