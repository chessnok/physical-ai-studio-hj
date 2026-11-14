/** How the next state snapshots are labelled: what they show, for anomaly detection, and a note. */
export interface SnapshotLabel {
    defective: boolean;
    comment: string;
}

/** The backend's limit (schemas/state_snapshot.py). */
export const COMMENT_MAX_LENGTH = 500;

export const NO_LABEL: SnapshotLabel = { defective: false, comment: '' };

const STORAGE_KEY_PREFIX = 'physicalai:snapshotLabel:';

export const getSnapshotLabelStorageKey = (projectId: string): string => `${STORAGE_KEY_PREFIX}${projectId}`;

/** The label last used in this browser for the project's snapshots; none if storage is unavailable. */
export const readSnapshotLabel = (projectId: string): SnapshotLabel => {
    try {
        const raw = window.localStorage.getItem(getSnapshotLabelStorageKey(projectId));
        if (raw === null) {
            return NO_LABEL;
        }
        const stored: unknown = JSON.parse(raw);
        if (typeof stored !== 'object' || stored === null) {
            return NO_LABEL;
        }
        const { defective, comment } = stored as Record<string, unknown>;
        return {
            defective: defective === true,
            comment: typeof comment === 'string' ? comment.slice(0, COMMENT_MAX_LENGTH) : '',
        };
    } catch {
        // Private browsing, blocked storage or a corrupt value: start unlabelled.
        return NO_LABEL;
    }
};

export const writeSnapshotLabel = (projectId: string, label: SnapshotLabel): void => {
    try {
        window.localStorage.setItem(getSnapshotLabelStorageKey(projectId), JSON.stringify(label));
    } catch {}
};
