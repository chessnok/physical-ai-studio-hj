import { ReactNode } from 'react';

import { toast, ToastContainer } from '@geti-ui/ui';
import { afterEach, beforeEach, describe, expect, it, rs } from '@rstest/core';
import { act, fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import useWebSocket from 'react-use-websocket';

import { SchemaStateSnapshot } from '../../api/openapi-spec';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { useHotkey } from '../hotkeys/use-hotkey';
import { RuntimeSessionProvider } from './runtime-session-provider';
import { describeSnapshot, StateSnapshotProvider } from './state-snapshot';
import { StateSnapshotControls } from './state-snapshot-controls';
import { getSnapshotLabelStorageKey, readSnapshotLabel } from './state-snapshot-label';

rs.mock('react-use-websocket', () => ({
    default: rs.fn(() => ({ sendJsonMessage: rs.fn() })),
}));

const SNAPSHOT: SchemaStateSnapshot = {
    snapshot_id: '20261113-171500-123',
    path: '/storage/state_snapshots/project-1/20261113-171500-123',
    captured_at: '2026-11-13T17:15:00.123+01:00',
    session_id: 'session-1',
    environment_name: 'Bench',
    cameras: [
        { name: 'Top', key: 'top', files: ['top.color.png', 'top.depth.png'], color: true, depth: true },
        { name: 'Arm', key: 'arm', files: ['arm.color.png'], color: true, depth: false },
    ],
    follower: true,
    leader: true,
    warnings: [],
    defective: false,
    comment: '',
};

const mockSettings = (bindings: Record<string, string> = {}) =>
    server.use(
        http.get('/api/settings', () =>
            HttpResponse.json({
                video_recording: { preset: null },
                trainer: {
                    request_timeout_s: 30,
                    download_read_timeout_s: 120,
                    stream_reconnect_max_s: 900,
                    stream_reconnect_backoff_max_s: 30,
                },
                huggingface: { hf_token: null },
                ssh: {
                    connect_timeout_s: 10,
                    command_timeout_s: 15,
                    preflight_timeout_s: 30,
                    image_pull_timeout_s: 1800,
                    trainer_shm_size_gb: 32,
                },
                hotkeys: { bindings },
            })
        )
    );

type CaptureRequest = { url: URL; body: unknown };

/** Answers captures with `respond`, or with the snapshot as labelled; returns what was asked. */
const mockCapture = (respond?: () => Response) => {
    const requests: CaptureRequest[] = [];
    server.use(
        http.post('/api/projects/{project_id}/state_snapshots', async ({ request }) => {
            const body = (await request.json().catch(() => null)) as Record<string, unknown> | null;
            requests.push({ url: new URL(request.url), body });
            return (respond?.() ?? HttpResponse.json({ ...SNAPSHOT, ...body }, { status: 201 })) as never;
        })
    );
    return requests;
};

const pressSpace = (target: Element | Window = window) => fireEvent.keyDown(target, { key: ' ' });
const settle = () => new Promise((resolve) => setTimeout(resolve, 50));

const Controls = ({
    project_id = 'project-1',
    session_id = 'session-1',
    layout,
    children,
}: {
    project_id?: string;
    session_id?: string | null;
    layout?: 'column' | 'row';
    children?: ReactNode;
}) => (
    <StateSnapshotProvider project_id={project_id} session_id={session_id}>
        <StateSnapshotControls layout={layout} />
        {children}
    </StateSnapshotProvider>
);

/** The recording page's hotkeys, as the base branch binds them. */
const RecordingHotkeys = ({ start, discard }: { start: () => void; discard: () => void }) => {
    useHotkey('Enter', start);
    useHotkey('Escape', discard);
    return null;
};

const defectiveSwitch = () => screen.getByRole('switch', { name: 'Defective' });
const commentField = () => screen.getByRole('textbox', { name: 'Comment' });

beforeEach(() => {
    window.localStorage.clear();
});

afterEach(() => {
    rs.restoreAllMocks();
});

describe('StateSnapshotControls', () => {
    it('captures on Space with the label and comment, and keeps them for the next snapshot', async () => {
        mockSettings();
        const requests = mockCapture();
        const notify = rs.spyOn(toast, 'positive').mockReturnValue(rs.fn());
        const user = userEvent.setup();
        render(<Controls />);

        await user.click(defectiveSwitch());
        await user.type(commentField(), '  scratch, "left" ');
        expect(requests).toHaveLength(0); // the spaces typed are text
        await user.keyboard('{Enter}'); // leaves the field
        expect(commentField()).not.toHaveFocus();
        await user.keyboard(' ');

        await waitFor(() => expect(notify).toHaveBeenCalledTimes(1));
        expect(requests).toHaveLength(1);
        expect(requests[0].url.pathname).toBe('/api/projects/project-1/state_snapshots');
        expect(requests[0].url.searchParams.get('session_id')).toBe('session-1');
        expect(requests[0].body).toEqual({ defective: true, comment: 'scratch, "left"' });
        const message = notify.mock.calls[0][0] as string;
        expect(message).toContain(`Defective snapshot "scratch, "left"" saved to ${SNAPSHOT.path}`);
        expect(message).toContain('Top (color, depth)');

        // A series of one class: the label stays.
        expect(defectiveSwitch()).toBeChecked();
        expect(commentField()).toHaveValue('  scratch, "left" ');
        await user.click(screen.getByRole('button', { name: 'Capture' }));
        await waitFor(() => expect(requests).toHaveLength(2));
        expect(requests[1].body).toEqual({ defective: true, comment: 'scratch, "left"' });
    });

    it('captures on Space on its own switch and button instead of flipping or pressing them', async () => {
        mockSettings();
        const requests = mockCapture();
        const notify = rs.spyOn(toast, 'positive').mockReturnValue(rs.fn());
        const user = userEvent.setup();
        render(<Controls />);

        await user.click(defectiveSwitch());
        defectiveSwitch().focus();
        await user.keyboard(' ');
        await waitFor(() => expect(notify).toHaveBeenCalledTimes(1));
        expect(defectiveSwitch()).toBeChecked();
        expect(requests.map((request) => request.body)).toEqual([{ defective: true, comment: '' }]);

        screen.getByRole('button', { name: 'Capture' }).focus();
        await user.keyboard(' ');
        await waitFor(() => expect(notify).toHaveBeenCalledTimes(2));
        await settle();
        expect(requests).toHaveLength(2); // once, not by the hotkey and the button press both
    });

    it('leaves Space to the comment field while typing; Escape leaves the field', async () => {
        mockSettings();
        const requests = mockCapture();
        rs.spyOn(toast, 'positive').mockReturnValue(rs.fn());
        const user = userEvent.setup();
        render(<Controls layout='row' />);

        await user.type(commentField(), 'a b c');
        pressSpace(commentField());
        await settle();
        expect(requests).toHaveLength(0);
        expect(commentField()).toHaveValue('a b c');

        await user.keyboard('{Escape}');
        expect(commentField()).not.toHaveFocus();
        await user.keyboard(' ');
        await waitFor(() => expect(requests).toHaveLength(1));
        expect(requests[0].body).toEqual({ defective: false, comment: 'a b c' });
    });

    it('keeps Enter and Escape typed into the comment from the recording hotkeys', async () => {
        mockSettings();
        const requests = mockCapture();
        const startEpisode = rs.fn();
        const discardEpisode = rs.fn();
        const user = userEvent.setup();
        render(
            <Controls>
                <RecordingHotkeys start={startEpisode} discard={discardEpisode} />
            </Controls>
        );

        await user.type(commentField(), 'dent{Enter}');
        expect(commentField()).not.toHaveFocus();
        await user.type(commentField(), ' left{Escape}');
        expect(commentField()).not.toHaveFocus();
        await settle();

        expect(startEpisode).not.toHaveBeenCalled();
        expect(discardEpisode).not.toHaveBeenCalled();
        expect(requests).toHaveLength(0);
        expect(commentField()).toHaveValue('dent left');
        // Outside the field the page's hotkeys work as before.
        await user.keyboard('{Enter}');
        expect(startEpisode).toHaveBeenCalledTimes(1);
    });

    it('gives focus back to the page after a click, so that Enter and Escape reach the page', async () => {
        mockSettings();
        const requests = mockCapture();
        const notify = rs.spyOn(toast, 'positive').mockReturnValue(rs.fn());
        server.use(http.get('/api/projects/{project_id}/state_snapshots', () => HttpResponse.json([])));
        const startEpisode = rs.fn();
        const discardEpisode = rs.fn();
        const user = userEvent.setup();
        render(
            <Controls>
                <RecordingHotkeys start={startEpisode} discard={discardEpisode} />
            </Controls>
        );

        await user.click(defectiveSwitch());
        expect(defectiveSwitch()).toBeChecked();
        expect(defectiveSwitch()).not.toHaveFocus();
        await user.keyboard('{Enter}');
        expect(startEpisode).toHaveBeenCalledTimes(1);

        await user.click(screen.getByRole('button', { name: 'Capture' }));
        await waitFor(() => expect(notify).toHaveBeenCalledTimes(1));
        await user.keyboard('{Enter}');
        expect(startEpisode).toHaveBeenCalledTimes(2);
        expect(requests).toHaveLength(1);

        await user.click(screen.getByRole('button', { name: 'Recent' }));
        await screen.findByText('No snapshots yet.');
        await user.keyboard('{Escape}'); // closes the list, and only that
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        expect(discardEpisode).not.toHaveBeenCalled();
        await waitFor(() => expect(screen.getByRole('button', { name: 'Recent' })).not.toHaveFocus());
        await user.keyboard('{Enter}');
        expect(startEpisode).toHaveBeenCalledTimes(3);
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });

    it('leaves Enter and Escape to the page while the "saved" toast is on screen', async () => {
        mockSettings();
        const requests = mockCapture();
        const shown = rs.spyOn(toast, 'positive'); // the real toast, an alertdialog that is not modal
        const startEpisode = rs.fn();
        const discardEpisode = rs.fn();
        const user = userEvent.setup();
        render(
            <>
                <ToastContainer />
                <Controls>
                    <RecordingHotkeys start={startEpisode} discard={discardEpisode} />
                </Controls>
            </>
        );

        pressSpace();
        await screen.findByText(/^Good snapshot saved to /);
        expect(requests).toHaveLength(1);
        expect(screen.getByRole('alertdialog')).toBeInTheDocument();

        await user.keyboard('{Enter}');
        await user.keyboard('{Escape}');

        expect(startEpisode).toHaveBeenCalledTimes(1);
        expect(discardEpisode).toHaveBeenCalledTimes(1);
        act(() => shown.mock.results.forEach((result) => (result.value as () => void)()));
    });

    it('remembers the label per project in this browser', async () => {
        mockSettings();
        const user = userEvent.setup();
        const first = render(<Controls project_id='project-1' />);
        await user.click(defectiveSwitch());
        await user.type(commentField(), 'dent');
        first.unmount();

        expect(JSON.parse(window.localStorage.getItem(getSnapshotLabelStorageKey('project-1')) ?? '')).toEqual({
            defective: true,
            comment: 'dent',
        });
        const again = render(<Controls project_id='project-1' />);
        expect(defectiveSwitch()).toBeChecked();
        expect(commentField()).toHaveValue('dent');
        again.unmount();

        render(<Controls project_id='project-2' />);
        expect(defectiveSwitch()).not.toBeChecked();
        expect(commentField()).toHaveValue('');
    });

    it('works without browser storage', async () => {
        mockSettings();
        const requests = mockCapture();
        rs.spyOn(toast, 'positive').mockReturnValue(rs.fn());
        const denied = () => {
            throw new DOMException('The operation is insecure.', 'SecurityError');
        };
        rs.spyOn(Storage.prototype, 'getItem').mockImplementation(denied);
        rs.spyOn(Storage.prototype, 'setItem').mockImplementation(denied);
        const user = userEvent.setup();
        render(<Controls />);

        await user.click(defectiveSwitch());
        pressSpace();

        await waitFor(() => expect(requests).toHaveLength(1));
        expect(requests[0].body).toEqual({ defective: true, comment: '' });
    });

    it('shows nothing and ignores the hotkey without a live session', async () => {
        mockSettings();
        const requests = mockCapture();
        render(<Controls session_id={null} />);

        pressSpace();
        await settle();

        expect(screen.queryByRole('switch', { name: 'Defective' })).not.toBeInTheDocument();
        expect(requests).toHaveLength(0);
    });

    it('says clearly why no snapshot was saved', async () => {
        mockSettings();
        mockCapture(() =>
            HttpResponse.json(
                {
                    message: 'No live session with an environment to snapshot.',
                    error_code: 'no_live_session',
                },
                { status: 409 }
            )
        );
        const notify = rs.spyOn(toast, 'negative').mockReturnValue(rs.fn());
        render(<Controls />);

        pressSpace();

        await waitFor(() => expect(notify).toHaveBeenCalled());
        expect(notify.mock.calls[0][0]).toContain('No live session with an environment to snapshot.');
    });

    it('leaves Space to a focused button and to text fields elsewhere on the page', async () => {
        mockSettings();
        const requests = mockCapture();
        render(
            <Controls>
                <button type='button'>Teleoperate</button>
                <input aria-label='Task' />
            </Controls>
        );

        pressSpace(screen.getByRole('button', { name: 'Teleoperate' }));
        pressSpace(screen.getByRole('textbox', { name: 'Task' }));
        await settle();

        expect(requests).toHaveLength(0);
    });

    it('follows a rebinding from the settings', async () => {
        mockSettings({ 'session.capture_snapshot': 'S' });
        const requests = mockCapture();
        rs.spyOn(toast, 'positive').mockReturnValue(rs.fn());
        render(<Controls />);
        await screen.findByText('S captures with this label, which stays for the next snapshots.');

        fireEvent.keyDown(window, { key: 's' });
        await waitFor(() => expect(requests).toHaveLength(1));
        pressSpace();
        await settle();

        expect(requests).toHaveLength(1);
    });
});

/** Serves the recent list from `initial` and applies PATCHes to it (or fails them); returns what was asked. */
const mockRecent = (
    initial: SchemaStateSnapshot[],
    { failPatches = false, csvWarning = null as string | null } = {}
) => {
    let listed = initial;
    const listRequests: URL[] = [];
    const patches: { snapshot_id: string; body: unknown }[] = [];
    server.use(
        http.get('/api/projects/{project_id}/state_snapshots', ({ request }) => {
            listRequests.push(new URL(request.url));
            return HttpResponse.json(listed);
        }),
        http.patch('/api/projects/{project_id}/state_snapshots/{snapshot_id}', async ({ request, params }) => {
            const body = (await request.json()) as Partial<SchemaStateSnapshot>;
            patches.push({ snapshot_id: String(params.snapshot_id), body });
            if (failPatches) {
                return HttpResponse.json(
                    { message: 'No space left on device', error_code: 'internal_error' },
                    { status: 500 }
                ) as never;
            }
            listed = listed.map((item) => (item.snapshot_id === params.snapshot_id ? { ...item, ...body } : item));
            const changed = listed.find((item) => item.snapshot_id === params.snapshot_id) ?? listed[0];
            return HttpResponse.json({ ...changed, csv_warning: csvWarning });
        })
    );
    return { listRequests, patches };
};

const openRecent = async (user: ReturnType<typeof userEvent.setup>) => {
    await user.click(screen.getByRole('button', { name: 'Recent' }));
    return screen.findByRole('dialog');
};

describe('recent snapshots', () => {
    it('lists the latest, changes their labels and downloads the CSV', async () => {
        mockSettings();
        const requests = mockCapture();
        const { listRequests, patches } = mockRecent([{ ...SNAPSHOT, comment: 'old' }]);
        const open = rs.spyOn(window, 'open').mockReturnValue(null);
        const user = userEvent.setup();
        render(<Controls />);

        const dialog = await openRecent(user);
        const toggle = await within(dialog).findByRole('switch', { name: 'Defective, snapshot of 17:15:00.123' });
        expect(listRequests[0].searchParams.get('limit')).toBe('8');
        expect(toggle).not.toBeChecked();

        await user.click(toggle);
        await waitFor(() =>
            expect(patches).toEqual([{ snapshot_id: SNAPSHOT.snapshot_id, body: { defective: true } }])
        );
        await waitFor(() => expect(toggle).toBeChecked());

        const field = within(dialog).getByRole('textbox', { name: 'Comment, snapshot of 17:15:00.123' });
        expect(field).toHaveValue('old');
        await user.clear(field);
        await user.type(field, 'dent, fixed{Enter}');
        await waitFor(() => expect(patches).toHaveLength(2));
        expect(patches[1]).toEqual({ snapshot_id: SNAPSHOT.snapshot_id, body: { comment: 'dent, fixed' } });
        expect(requests).toHaveLength(0); // typing a space there captured nothing

        await user.click(within(dialog).getByRole('button', { name: 'Download CSV of all snapshots' }));
        expect(open).toHaveBeenCalledWith(
            expect.stringContaining('/api/projects/project-1/state_snapshots/snapshots.csv'),
            '_blank',
            'noopener,noreferrer'
        );
    });

    it('captures on Space after a switch in the list was toggled, without flipping it back', async () => {
        mockSettings();
        const requests = mockCapture();
        const notify = rs.spyOn(toast, 'positive').mockReturnValue(rs.fn());
        const { patches } = mockRecent([SNAPSHOT]);
        const user = userEvent.setup();
        render(<Controls />);

        const dialog = await openRecent(user);
        const toggle = await within(dialog).findByRole('switch', { name: 'Defective, snapshot of 17:15:00.123' });
        await user.click(toggle);
        await waitFor(() => expect(toggle).toBeChecked());
        expect(toggle).toHaveFocus(); // where a click leaves it, in the popover

        await user.keyboard(' ');

        await waitFor(() => expect(notify).toHaveBeenCalledTimes(1));
        expect(requests).toHaveLength(1);
        await settle();
        expect(patches).toEqual([{ snapshot_id: SNAPSHOT.snapshot_id, body: { defective: true } }]);
        expect(toggle).toBeChecked();
    });

    it('shows a snapshot from before labels as unlabelled, to label as good or defective', async () => {
        mockSettings();
        const { patches } = mockRecent([
            { ...SNAPSHOT, defective: null },
            { ...SNAPSHOT, snapshot_id: '20261113-171459-001', defective: null },
        ]);
        const user = userEvent.setup();
        render(<Controls />);

        const dialog = await openRecent(user);
        await within(dialog).findByRole('button', { name: 'Good, snapshot of 17:15:00.123' });
        expect(within(dialog).getAllByText('Unlabelled')).toHaveLength(2);
        expect(within(dialog).queryByRole('switch')).not.toBeInTheDocument();

        await user.click(within(dialog).getByRole('button', { name: 'Good, snapshot of 17:15:00.123' }));
        const good = await within(dialog).findByRole('switch', { name: 'Defective, snapshot of 17:15:00.123' });
        expect(good).not.toBeChecked();
        await user.click(within(dialog).getByRole('button', { name: 'Defective, snapshot of 17:14:59.001' }));
        const bad = await within(dialog).findByRole('switch', { name: 'Defective, snapshot of 17:14:59.001' });
        await waitFor(() => expect(bad).toBeChecked());

        expect(patches).toEqual([
            { snapshot_id: SNAPSHOT.snapshot_id, body: { defective: false } },
            { snapshot_id: '20261113-171459-001', body: { defective: true } },
        ]);
        expect(within(dialog).queryByText('Unlabelled')).not.toBeInTheDocument();
    });

    it('shows what the backend has after a change failed', async () => {
        mockSettings();
        const { listRequests, patches } = mockRecent([SNAPSHOT], { failPatches: true });
        const notify = rs.spyOn(toast, 'negative').mockReturnValue(rs.fn());
        const user = userEvent.setup();
        render(<Controls />);

        const dialog = await openRecent(user);
        const toggle = await within(dialog).findByRole('switch', { name: 'Defective, snapshot of 17:15:00.123' });
        expect(listRequests).toHaveLength(1);
        await user.click(toggle);

        await waitFor(() => expect(notify).toHaveBeenCalledTimes(1));
        expect(notify.mock.calls[0][0]).toContain('No space left on device');
        await waitFor(() => expect(listRequests).toHaveLength(2)); // fetched again
        expect(patches).toHaveLength(1);
        expect(toggle).not.toBeChecked();
    });

    it('says when a change is saved but the CSV is not rewritten yet', async () => {
        mockSettings();
        const warning =
            'The label is saved, but snapshots.csv could not be rewritten (No space left on device); ' +
            'the next capture or CSV download brings it up to date.';
        mockRecent([SNAPSHOT], { csvWarning: warning });
        const info = rs.spyOn(toast, 'info').mockReturnValue(rs.fn());
        const negative = rs.spyOn(toast, 'negative').mockReturnValue(rs.fn());
        const user = userEvent.setup();
        render(<Controls />);

        const dialog = await openRecent(user);
        const toggle = await within(dialog).findByRole('switch', { name: 'Defective, snapshot of 17:15:00.123' });
        await user.click(toggle);

        await waitFor(() => expect(info).toHaveBeenCalledWith(warning, expect.anything()));
        expect(toggle).toBeChecked();
        expect(negative).not.toHaveBeenCalled();
    });

    it('keeps Enter and Escape in the open list from the recording hotkeys; Escape drops a comment edit', async () => {
        mockSettings();
        const { patches } = mockRecent([{ ...SNAPSHOT, comment: 'old' }]);
        const startEpisode = rs.fn();
        const discardEpisode = rs.fn();
        const user = userEvent.setup();
        render(
            <Controls>
                <RecordingHotkeys start={startEpisode} discard={discardEpisode} />
            </Controls>
        );

        const dialog = await openRecent(user);
        const field = await within(dialog).findByRole('textbox', { name: 'Comment, snapshot of 17:15:00.123' });
        dialog.focus();
        await user.keyboard('{Enter}');
        expect(startEpisode).not.toHaveBeenCalled();

        await user.type(field, ' edited');
        await user.keyboard('{Escape}');
        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        await settle();

        expect(discardEpisode).not.toHaveBeenCalled();
        expect(patches).toEqual([]); // the edit was dropped, not saved
    });
});

describe('viewing and deleting a recent snapshot', () => {
    const WITH_DEPTH: SchemaStateSnapshot = {
        ...SNAPSHOT,
        defective: true,
        comment: 'dent',
        cameras: [
            {
                name: 'Top',
                key: 'top',
                files: ['top.color.png', 'top.depth.png', 'top.depth_preview.png'],
                color: true,
                depth: true,
            },
            { name: 'Arm', key: 'arm', files: ['arm.color.png'], color: true, depth: false },
        ],
    };

    it('shows a thumbnail, and all images with the label in a preview', async () => {
        mockSettings();
        mockRecent([WITH_DEPTH]);
        const user = userEvent.setup();
        render(<Controls />);

        const list = await openRecent(user);
        const thumbnail = await within(list).findByRole('button', { name: 'Show snapshot of 17:15:00.123' });
        expect(thumbnail.querySelector('img')?.getAttribute('src')).toContain(
            '/api/projects/project-1/state_snapshots/20261113-171500-123/files/top.color.png'
        );
        await user.click(thumbnail);

        const preview = await screen.findByRole('dialog', { name: 'Snapshot of 17:15:00.123' });
        expect(
            within(preview)
                .getAllByRole('img')
                .map((image) => image.getAttribute('alt'))
        ).toEqual(['Top, color', 'Top, depth', 'Arm, color']);
        expect(within(preview).getByRole('img', { name: 'Top, depth' }).getAttribute('src')).toContain(
            '/files/top.depth_preview.png'
        );
        expect(within(preview).getByText(/^Defective, "dent"\. Captured /)).toBeInTheDocument();

        await user.click(within(preview).getByRole('button', { name: 'Close' }));
        await waitFor(() =>
            expect(screen.queryByRole('dialog', { name: 'Snapshot of 17:15:00.123' })).not.toBeInTheDocument()
        );
        expect(screen.getByRole('switch', { name: 'Defective, snapshot of 17:15:00.123' })).toBeInTheDocument();
    });

    it('moves a snapshot to the trash after a confirmation that keeps Escape from the page', async () => {
        mockSettings();
        const { listRequests } = mockRecent([WITH_DEPTH]);
        const deletes: string[] = [];
        server.use(
            http.delete('/api/projects/{project_id}/state_snapshots/{snapshot_id}', ({ params }) => {
                deletes.push(String(params.snapshot_id));
                server.use(http.get('/api/projects/{project_id}/state_snapshots', () => HttpResponse.json([])));
                return HttpResponse.json({
                    snapshot_id: String(params.snapshot_id),
                    moved_to: `/storage/state_snapshots_trash/20261114-120000/project-1/${params.snapshot_id}`,
                    csv_warning: null,
                });
            })
        );
        const notify = rs.spyOn(toast, 'positive').mockReturnValue(rs.fn());
        const startEpisode = rs.fn();
        const discardEpisode = rs.fn();
        const user = userEvent.setup();
        render(
            <Controls>
                <RecordingHotkeys start={startEpisode} discard={discardEpisode} />
            </Controls>
        );

        const list = await openRecent(user);
        const trash = await within(list).findByRole('button', { name: 'Move snapshot of 17:15:00.123 to trash' });
        await user.click(trash);
        await screen.findByRole('alertdialog');
        await user.keyboard('{Escape}');
        await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument());
        expect(discardEpisode).not.toHaveBeenCalled();
        expect(deletes).toEqual([]);

        await user.click(trash);
        const confirm = await screen.findByRole('alertdialog');
        await user.click(within(confirm).getByRole('button', { name: 'Move to trash' }));

        await waitFor(() => expect(notify).toHaveBeenCalledTimes(1));
        expect(deletes).toEqual([SNAPSHOT.snapshot_id]);
        expect(notify.mock.calls[0][0]).toContain('moved to trash: /storage/state_snapshots_trash/');
        await within(list).findByText('No snapshots yet.');
        expect(listRequests.length).toBeGreaterThanOrEqual(1);
        expect(startEpisode).not.toHaveBeenCalled();
    });
});

describe('readSnapshotLabel', () => {
    it('starts unlabelled from nothing or from a value it does not understand', () => {
        expect(readSnapshotLabel('project-1')).toEqual({ defective: false, comment: '' });
        window.localStorage.setItem(getSnapshotLabelStorageKey('project-1'), '{not json');
        expect(readSnapshotLabel('project-1')).toEqual({ defective: false, comment: '' });
        window.localStorage.setItem(getSnapshotLabelStorageKey('project-1'), '{"defective":"yes","comment":5}');
        expect(readSnapshotLabel('project-1')).toEqual({ defective: false, comment: '' });
        window.localStorage.setItem(
            getSnapshotLabelStorageKey('project-1'),
            JSON.stringify({ defective: true, comment: 'x'.repeat(600) })
        );
        expect(readSnapshotLabel('project-1')).toEqual({ defective: true, comment: 'x'.repeat(500) });
    });
});

describe('describeSnapshot', () => {
    it('says how the snapshot is labelled', () => {
        expect(describeSnapshot(SNAPSHOT)).toMatch(/^Good snapshot saved to /);
        expect(describeSnapshot({ ...SNAPSHOT, defective: null })).toMatch(/^Unlabelled snapshot saved to /);
        expect(describeSnapshot({ ...SNAPSHOT, defective: true, comment: 'dent' })).toMatch(
            /^Defective snapshot "dent" saved to /
        );
    });

    it('reports what could not be captured', () => {
        const message = describeSnapshot({
            ...SNAPSHOT,
            cameras: [],
            leader: false,
            warnings: ["The leader's joints are known only while teleoperation runs"],
        });

        expect(message).toContain('no camera frames; follower joints.');
        expect(message).toContain('known only while teleoperation runs');
    });
});

describe('in a live session', () => {
    it('is on once the session has an environment, for that session', async () => {
        mockSettings();
        const requests = mockCapture();
        rs.spyOn(toast, 'positive').mockReturnValue(rs.fn());
        render(
            <RuntimeSessionProvider project_id='project-1' environment={null}>
                <StateSnapshotControls />
            </RuntimeSessionProvider>
        );
        const onMessage = rs.mocked(useWebSocket).mock.calls.at(-1)?.[1]?.onMessage;
        const state = (environment: object | null) =>
            act(() => {
                onMessage?.(
                    new MessageEvent('message', {
                        data: JSON.stringify({ type: 'state', data: { session_id: 'session-9', environment } }),
                    })
                );
            });

        state(null);
        pressSpace();
        await settle();
        expect(requests).toHaveLength(0);
        expect(screen.queryByRole('switch', { name: 'Defective' })).not.toBeInTheDocument();

        state({
            environment_id: null,
            name: 'Bench',
            robot_id: 'robot-1',
            leader_id: null,
            joint_names: ['gripper.pos'],
            cameras: [],
        });
        await screen.findByRole('switch', { name: 'Defective' });
        pressSpace();

        await waitFor(() => expect(requests).toHaveLength(1));
        expect(requests[0].url.searchParams.get('session_id')).toBe('session-9');
    });
});
