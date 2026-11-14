import { afterEach, describe, expect, it, rs } from '@rstest/core';
import { fireEvent, renderHook } from '@testing-library/react';

import { HOTKEY_ACTIONS } from './hotkey-actions';
import { useHotkey } from './use-hotkey';

describe('recording hotkey defaults', () => {
    it('start and accept on Enter, discard on Escape', () => {
        expect(HOTKEY_ACTIONS['recording.start_episode'].defaultCombo).toBe('Enter');
        expect(HOTKEY_ACTIONS['recording.accept_episode'].defaultCombo).toBe('Enter');
        expect(HOTKEY_ACTIONS['recording.discard_episode'].defaultCombo).toBe('Escape');
    });
});

describe('useHotkey with Enter and Escape', () => {
    afterEach(() => {
        document.body.innerHTML = '';
    });

    it('fires on Enter pressed on the page', () => {
        const handler = rs.fn();
        renderHook(() => useHotkey('Enter', handler));
        fireEvent.keyDown(document.body, { key: 'Enter' });
        expect(handler).toHaveBeenCalledTimes(1);
    });

    it('leaves Enter on a focused button to the button', () => {
        const handler = rs.fn();
        renderHook(() => useHotkey('Enter', handler));
        const button = document.createElement('button');
        document.body.appendChild(button);
        fireEvent.keyDown(button, { key: 'Enter' });
        expect(handler).not.toHaveBeenCalled();
    });

    it('ignores Enter typed into a text field', () => {
        const handler = rs.fn();
        renderHook(() => useHotkey('Enter', handler));
        const input = document.createElement('input');
        document.body.appendChild(input);
        fireEvent.keyDown(input, { key: 'Enter' });
        expect(handler).not.toHaveBeenCalled();
    });

    it('fires on Escape when no overlay is open', () => {
        const handler = rs.fn();
        renderHook(() => useHotkey('Escape', handler));
        fireEvent.keyDown(document.body, { key: 'Escape' });
        expect(handler).toHaveBeenCalledTimes(1);
    });

    it('leaves Escape to an open dialog', () => {
        const handler = rs.fn();
        renderHook(() => useHotkey('Escape', handler));
        const dialog = document.createElement('div');
        dialog.setAttribute('role', 'dialog');
        document.body.appendChild(dialog);
        fireEvent.keyDown(document.body, { key: 'Escape' });
        expect(handler).not.toHaveBeenCalled();
    });

    it('ignores a key another handler already handled', () => {
        const handler = rs.fn();
        renderHook(() => useHotkey('Escape', handler));
        const target = document.createElement('div');
        document.body.appendChild(target);
        target.addEventListener('keydown', (e) => e.preventDefault());
        fireEvent.keyDown(target, { key: 'Escape' });
        expect(handler).not.toHaveBeenCalled();
    });
});
