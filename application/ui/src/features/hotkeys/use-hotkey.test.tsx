import {
    ActionButton,
    Content,
    Dialog,
    DialogTrigger,
    Heading,
    Item,
    Menu,
    MenuTrigger,
    toast,
    ToastContainer,
} from '@geti-ui/ui';
import { afterEach, describe, expect, it, rs } from '@rstest/core';
import { act, fireEvent, renderHook, screen } from '@testing-library/react';

import { render } from '../../test-utils/render';
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

    it('leaves Enter to an open dialog, menu or listbox', () => {
        const handler = rs.fn();
        renderHook(() => useHotkey('Enter', handler));
        const overlay = document.createElement('div');
        document.body.appendChild(overlay);

        for (const role of ['dialog', 'alertdialog', 'menu', 'listbox']) {
            overlay.setAttribute('role', role);
            fireEvent.keyDown(document.body, { key: 'Enter' });
        }
        expect(handler).not.toHaveBeenCalled();

        overlay.remove();
        fireEvent.keyDown(document.body, { key: 'Enter' });
        expect(handler).toHaveBeenCalledTimes(1);
    });

    it('still fires Space with a dialog open', () => {
        const handler = rs.fn();
        renderHook(() => useHotkey(' ', handler));
        const dialog = document.createElement('div');
        dialog.setAttribute('role', 'dialog');
        document.body.appendChild(dialog);
        fireEvent.keyDown(dialog, { key: ' ' });
        expect(handler).toHaveBeenCalledTimes(1);
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

const PageHotkeys = ({ enter, escape }: { enter: () => void; escape: () => void }) => {
    useHotkey('Enter', enter);
    useHotkey('Escape', escape);
    return null;
};

describe('useHotkey with notifications and overlays on screen', () => {
    const closeToasts: (() => void)[] = [];

    afterEach(() => {
        act(() => closeToasts.splice(0).forEach((close) => close()));
    });

    /** A real toast, as a snapshot capture shows one: an alertdialog, but not a modal one. */
    const showToast = async (message: string) => {
        act(() => {
            closeToasts.push(toast.positive(message));
        });
        await screen.findByText(message);
        expect(screen.getByRole('alertdialog')).toHaveAttribute('aria-modal', 'false');
    };

    it('fires Enter and Escape while a toast is on screen', async () => {
        const enter = rs.fn();
        const escape = rs.fn();
        render(
            <>
                <ToastContainer />
                <PageHotkeys enter={enter} escape={escape} />
            </>
        );
        await showToast('Good snapshot saved');

        fireEvent.keyDown(document.body, { key: 'Enter' });
        fireEvent.keyDown(document.body, { key: 'Escape' });

        expect(enter).toHaveBeenCalledTimes(1);
        expect(escape).toHaveBeenCalledTimes(1);
    });

    it('leaves Enter and Escape to a modal dialog, also with a toast on screen', async () => {
        const enter = rs.fn();
        const escape = rs.fn();
        render(
            <>
                <ToastContainer />
                <PageHotkeys enter={enter} escape={escape} />
                <DialogTrigger type='modal' defaultOpen>
                    <ActionButton>Open</ActionButton>
                    <Dialog>
                        <Heading>Discard the episode?</Heading>
                        <Content>It is not saved.</Content>
                    </Dialog>
                </DialogTrigger>
            </>
        );
        await screen.findByRole('dialog');
        await showToast('Good snapshot saved');

        fireEvent.keyDown(document.body, { key: 'Enter' });
        fireEvent.keyDown(document.body, { key: 'Escape' });

        expect(enter).not.toHaveBeenCalled();
        expect(escape).not.toHaveBeenCalled();
    });

    it('leaves Enter and Escape to an open menu', async () => {
        const enter = rs.fn();
        const escape = rs.fn();
        render(
            <>
                <PageHotkeys enter={enter} escape={escape} />
                <MenuTrigger defaultOpen>
                    <ActionButton>Actions</ActionButton>
                    <Menu>
                        <Item key='rename'>Rename</Item>
                    </Menu>
                </MenuTrigger>
            </>
        );
        await screen.findByRole('menu');

        fireEvent.keyDown(document.body, { key: 'Enter' });
        fireEvent.keyDown(document.body, { key: 'Escape' });

        expect(enter).not.toHaveBeenCalled();
        expect(escape).not.toHaveBeenCalled();
    });
});
