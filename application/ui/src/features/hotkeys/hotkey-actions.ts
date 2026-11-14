export type HotkeyActionId =
    'recording.start_episode' | 'recording.accept_episode' | 'recording.discard_episode' | 'session.capture_snapshot';

type HotkeyActionDefinition = {
    label: string;
    scope: string;
    defaultCombo: string;
};

export const HOTKEY_ACTIONS: Record<HotkeyActionId, HotkeyActionDefinition> = {
    'recording.start_episode': {
        label: 'Start episode',
        scope: 'Recording',
        defaultCombo: 'Enter',
    },
    'recording.accept_episode': {
        label: 'Accept episode',
        scope: 'Recording',
        defaultCombo: 'Enter',
    },
    'recording.discard_episode': {
        label: 'Discard episode',
        scope: 'Recording',
        defaultCombo: 'Escape',
    },
    // Read-only, so it may fire anywhere a session is live, recording or not.
    'session.capture_snapshot': {
        label: 'Capture state snapshot',
        scope: 'Live session',
        defaultCombo: ' ',
    },
};

export const HOTKEY_ACTION_IDS = Object.keys(HOTKEY_ACTIONS) as HotkeyActionId[];
