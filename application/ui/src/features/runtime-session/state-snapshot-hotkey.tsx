import { useCallback } from 'react';

import { toast } from '@geti-ui/ui';

import { $api } from '../../api/client';
import { getApiErrorMessage } from '../../api/errors';
import { SchemaStateSnapshot } from '../../api/openapi-spec';
import { getEffectiveBindings } from '../hotkeys/key-combo';
import { useHotkey } from '../hotkeys/use-hotkey';

const SNAPSHOT_TOAST = { timeout: 8_000 };
const SNAPSHOT_ERROR_TOAST = { timeout: 10_000 };

/** "Saved to …: top (color, depth), arm (color); follower and leader joints." */
export const describeSnapshot = (snapshot: SchemaStateSnapshot): string => {
    const cameras = snapshot.cameras
        .filter((camera) => camera.color || camera.depth)
        .map(
            (camera) =>
                `${camera.name} (${[camera.color && 'color', camera.depth && 'depth'].filter(Boolean).join(', ')})`
        );
    const robots = [snapshot.follower && 'follower', snapshot.leader && 'leader'].filter(Boolean).join(' and ');
    const parts = [
        cameras.length > 0 ? cameras.join(', ') : 'no camera frames',
        robots !== '' ? `${robots} joints` : 'no joints',
    ];
    const warnings = (snapshot.warnings ?? []).length > 0 ? ` ${(snapshot.warnings ?? []).join(' ')}` : '';
    return `Snapshot saved to ${snapshot.path}: ${parts.join('; ')}.${warnings}`;
};

/**
 * Saves a state snapshot of the live session ``session_id`` when its hotkey (Space by default) is pressed.
 *
 * The session provider mounts it while its session has an environment, so it works wherever one is
 * live: the environment view, recording, inference. Snapshots are read-only on the backend.
 */
export const StateSnapshotHotkey = ({ project_id, session_id }: { project_id: string; session_id: string }) => {
    const { data: settings } = $api.useQuery('get', '/api/settings');
    const combo = getEffectiveBindings(settings?.hotkeys.bindings)['session.capture_snapshot'];
    const { mutate, isPending } = $api.useMutation('post', '/api/projects/{project_id}/state_snapshots');

    const takeSnapshot = useCallback(() => {
        if (isPending) {
            return;
        }
        mutate(
            { params: { path: { project_id }, query: { session_id } } },
            {
                onSuccess: (snapshot) => {
                    toast.positive(describeSnapshot(snapshot), SNAPSHOT_TOAST);
                },
                onError: (error) => {
                    toast.negative(
                        `Could not save a snapshot: ${getApiErrorMessage(error) ?? 'the backend did not answer.'}`,
                        SNAPSHOT_ERROR_TOAST
                    );
                },
            }
        );
    }, [isPending, mutate, project_id, session_id]);

    useHotkey(combo, takeSnapshot);

    return null;
};
