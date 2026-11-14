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

// Enter activates the focused button, link, menu item, option or tab; leave it to that control.
const ENTER_ACTIVATED_ROLES = new Set(['button', 'link', 'menuitem', 'option', 'tab']);

const isEnterActivatedTarget = (target: EventTarget | null): boolean => {
    if (!(target instanceof HTMLElement)) {
        return false;
    }

    const role = target.getAttribute('role');
    return (
        target instanceof HTMLButtonElement ||
        target instanceof HTMLAnchorElement ||
        (role !== null && ENTER_ACTIVATED_ROLES.has(role))
    );
};

// Escape closes an open dialog, menu or listbox first; it must not also act on the page behind it.
const hasOpenOverlay = (): boolean =>
    document.querySelector('[role="dialog"], [role="alertdialog"], [role="menu"], [role="listbox"]') !== null;

/**
 * Fires `handler` when `combo` (a canonical string from `keyboardEventToCombo`) is
 * pressed anywhere in the document, while `enabled`. Ignores keystrokes into text
 * inputs/content-editable elements, key-repeat events, events another handler already
 * handled, Space or Enter on a focused control that the key activates, and Escape
 * while a dialog, menu or listbox is open.
 */
export const useHotkey = (combo: string, handler: () => void, enabled = true): void => {
    useEffect(() => {
        if (!enabled) {
            return;
        }

        const onKeyDown = (e: KeyboardEvent) => {
            if (
                e.repeat ||
                e.defaultPrevented ||
                isTypingTarget(e.target) ||
                (e.key === ' ' && isSpaceActivatedTarget(e.target)) ||
                (e.key === 'Enter' && isEnterActivatedTarget(e.target)) ||
                (e.key === 'Escape' && hasOpenOverlay())
            ) {
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
