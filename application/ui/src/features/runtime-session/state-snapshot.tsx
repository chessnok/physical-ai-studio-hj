import { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useState } from 'react';

import { toast } from '@geti-ui/ui';

import { $api } from '../../api/client';
import { getApiErrorMessage } from '../../api/errors';
import { SchemaStateSnapshot } from '../../api/openapi-spec';
import { getQueryKey } from '../../query-client/query-client.interface';
import { getEffectiveBindings } from '../hotkeys/key-combo';
import { useHotkey } from '../hotkeys/use-hotkey';
import { readSnapshotLabel, SnapshotLabel, writeSnapshotLabel } from './state-snapshot-label';

const SNAPSHOT_TOAST = { timeout: 8_000 };
const SNAPSHOT_ERROR_TOAST = { timeout: 10_000 };

/** 'Defective snapshot "scratch" saved to …: top (color, depth), arm (color); follower and leader joints.' */
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
    const kind = snapshot.defective ? 'Defective' : 'Good';
    const comment = snapshot.comment ? ` "${snapshot.comment}"` : '';
    return `${kind} snapshot${comment} saved to ${snapshot.path}: ${parts.join('; ')}.${warnings}`;
};

/** The list of a project's snapshots; captures and label changes refresh it. */
export const snapshotListKey = (project_id: string) =>
    getQueryKey(['get', '/api/projects/{project_id}/state_snapshots', { params: { path: { project_id } } }]);

interface StateSnapshotContextValue {
    project_id: string;
    /** The label the next snapshot gets; kept between captures and remembered per project in this browser. */
    label: SnapshotLabel;
    setLabel: (changes: Partial<SnapshotLabel>) => void;
    /** Saves a snapshot of the live session with the current label. */
    capture: () => void;
    isCapturing: boolean;
    /** The capture hotkey, as `keyboardEventToCombo` writes it. */
    combo: string;
}

const StateSnapshotContext = createContext<StateSnapshotContextValue | null>(null);

/** The live session's snapshot label and capture, or null where no session with an environment is live. */
export const useStateSnapshot = (): StateSnapshotContextValue | null => useContext(StateSnapshotContext);

/** The project's snapshot label, read from and written to this browser's storage. */
const useSnapshotLabel = (project_id: string) => {
    const [stored, setStored] = useState(() => ({ project_id, label: readSnapshotLabel(project_id) }));
    // Adjusted while rendering: another project has its own label.
    if (stored.project_id !== project_id) {
        setStored({ project_id, label: readSnapshotLabel(project_id) });
    }

    useEffect(() => {
        writeSnapshotLabel(stored.project_id, stored.label);
    }, [stored]);

    const setLabel = useCallback((changes: Partial<SnapshotLabel>) => {
        setStored((previous) => ({ ...previous, label: { ...previous.label, ...changes } }));
    }, []);

    return [stored.label, setLabel] as const;
};

/**
 * Saves state snapshots of the live session `session_id`, labelled as its controls say, when the
 * capture hotkey (Space by default) is pressed or the controls' Capture button.
 *
 * The session provider mounts it around its page, with `session_id` while its session has an
 * environment, so it works wherever one is live: the environment view, recording, inference.
 * Snapshots are read-only on the backend.
 */
export const StateSnapshotProvider = ({
    project_id,
    session_id,
    children,
}: {
    project_id: string;
    /** The live session with an environment; null turns capturing off. */
    session_id: string | null;
    children: ReactNode;
}) => {
    const { data: settings } = $api.useQuery('get', '/api/settings');
    const combo = getEffectiveBindings(settings?.hotkeys.bindings)['session.capture_snapshot'];
    const [label, setLabel] = useSnapshotLabel(project_id);
    const { mutate, isPending } = $api.useMutation('post', '/api/projects/{project_id}/state_snapshots', {
        meta: { invalidates: [snapshotListKey(project_id)] },
    });

    const capture = useCallback(() => {
        if (session_id === null || isPending) {
            return;
        }
        mutate(
            {
                params: { path: { project_id }, query: { session_id } },
                body: { defective: label.defective, comment: label.comment.trim() },
            },
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
    }, [isPending, label, mutate, project_id, session_id]);

    useHotkey(combo, capture, session_id !== null);

    const value = useMemo(
        () => (session_id === null ? null : { project_id, label, setLabel, capture, isCapturing: isPending, combo }),
        [project_id, session_id, label, setLabel, capture, isPending, combo]
    );

    return <StateSnapshotContext.Provider value={value}>{children}</StateSnapshotContext.Provider>;
};
