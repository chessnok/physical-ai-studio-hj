import { useEffect } from 'react';

import { keyboardEventToCombo } from './key-combo';

const isTypingTarget = (target: EventTarget | null): boolean => {
    if (!(target instanceof HTMLElement)) {
        return false;
    }

    return (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        target.isContentEditable
    );
};

// Space activates the focused button, switch, checkbox or option; leave it to that control.
const SPACE_ACTIVATED_ROLES = new Set(['button', 'checkbox', 'switch', 'radio', 'option', 'menuitem', 'tab', 'slider']);

const isSpaceActivatedTarget = (target: EventTarget | null): boolean => {
    if (!(target instanceof HTMLElement)) {
        return false;
    }

    const role = target.getAttribute('role');
    return target instanceof HTMLButtonElement || (role !== null && SPACE_ACTIVATED_ROLES.has(role));
};

/**
 * Fires `handler` when `combo` (a canonical string from `keyboardEventToCombo`) is
 * pressed anywhere in the document, while `enabled`. Ignores keystrokes into text
 * inputs/content-editable elements, key-repeat events, and Space on a focused
 * control that Space activates.
 */
export const useHotkey = (combo: string, handler: () => void, enabled = true): void => {
    useEffect(() => {
        if (!enabled) {
            return;
        }

        const onKeyDown = (e: KeyboardEvent) => {
            if (e.repeat || isTypingTarget(e.target) || (e.key === ' ' && isSpaceActivatedTarget(e.target))) {
                return;
            }

            if (keyboardEventToCombo(e) === combo) {
                e.preventDefault();
                handler();
            }
        };

        window.addEventListener('keydown', onKeyDown);
        return () => window.removeEventListener('keydown', onKeyDown);
    }, [combo, enabled, handler]);
};
