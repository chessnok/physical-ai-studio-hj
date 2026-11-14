import { toast } from '@geti-ui/ui';
import { afterEach, describe, expect, it, rs } from '@rstest/core';
import { act, fireEvent, screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import useWebSocket from 'react-use-websocket';

import { SchemaStateSnapshot } from '../../api/openapi-spec';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { render } from '../../test-utils/render';
import { RuntimeSessionProvider } from './runtime-session-provider';
import { describeSnapshot, StateSnapshotHotkey } from './state-snapshot-hotkey';

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

const mockCapture = (respond: () => Response) => {
    const requests: URL[] = [];
    server.use(
        http.post('/api/projects/{project_id}/state_snapshots', ({ request }) => {
            requests.push(new URL(request.url));
            return respond() as never;
        })
    );
    return requests;
};

const pressSpace = (target: Element | Window = window) => fireEvent.keyDown(target, { key: ' ' });

afterEach(() => {
    rs.restoreAllMocks();
});

describe('StateSnapshotHotkey', () => {
    it('saves a snapshot of the live session on Space and says what it saved', async () => {
        mockSettings();
        const requests = mockCapture(() => HttpResponse.json(SNAPSHOT, { status: 201 }));
        const notify = rs.spyOn(toast, 'positive').mockReturnValue(rs.fn());
        render(<StateSnapshotHotkey project_id='project-1' session_id='session-1' />);

        pressSpace();

        await waitFor(() => expect(notify).toHaveBeenCalled());
        expect(requests).toHaveLength(1);
        expect(requests[0].pathname).toBe('/api/projects/project-1/state_snapshots');
        expect(requests[0].searchParams.get('session_id')).toBe('session-1');
        const message = notify.mock.calls[0][0] as string;
        expect(message).toContain(SNAPSHOT.path);
        expect(message).toContain('Top (color, depth)');
        expect(message).toContain('follower and leader joints');
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
        render(<StateSnapshotHotkey project_id='project-1' session_id='session-1' />);

        pressSpace();

        await waitFor(() => expect(notify).toHaveBeenCalled());
        expect(notify.mock.calls[0][0]).toContain('No live session with an environment to snapshot.');
    });

    it('leaves Space to a focused button and to text fields', async () => {
        mockSettings();
        const requests = mockCapture(() => HttpResponse.json(SNAPSHOT, { status: 201 }));
        render(
            <>
                <StateSnapshotHotkey project_id='project-1' session_id='session-1' />
                <button type='button'>Teleoperate</button>
                <input aria-label='Task' />
            </>
        );

        pressSpace(screen.getByRole('button', { name: 'Teleoperate' }));
        pressSpace(screen.getByRole('textbox', { name: 'Task' }));
        await new Promise((resolve) => setTimeout(resolve, 50));

        expect(requests).toHaveLength(0);
    });

    it('follows a rebinding from the settings', async () => {
        mockSettings({ 'session.capture_snapshot': 'S' });
        const requests = mockCapture(() => HttpResponse.json(SNAPSHOT, { status: 201 }));
        rs.spyOn(toast, 'positive').mockReturnValue(rs.fn());
        render(<StateSnapshotHotkey project_id='project-1' session_id='session-1' />);
        await new Promise((resolve) => setTimeout(resolve, 100)); // the settings load

        fireEvent.keyDown(window, { key: 's' });
        await waitFor(() => expect(requests).toHaveLength(1));
        pressSpace();
        await new Promise((resolve) => setTimeout(resolve, 50));

        expect(requests).toHaveLength(1);
    });
});

describe('describeSnapshot', () => {
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
        const requests = mockCapture(() => HttpResponse.json(SNAPSHOT, { status: 201 }));
        rs.spyOn(toast, 'positive').mockReturnValue(rs.fn());
        render(
            <RuntimeSessionProvider project_id='project-1' environment={null}>
                {null}
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
        await new Promise((resolve) => setTimeout(resolve, 50));
        expect(requests).toHaveLength(0);

        state({
            environment_id: null,
            name: 'Bench',
            robot_id: 'robot-1',
            leader_id: null,
            joint_names: ['gripper.pos'],
            cameras: [],
        });
        await new Promise((resolve) => setTimeout(resolve, 100)); // the settings load
        pressSpace();

        await waitFor(() => expect(requests).toHaveLength(1));
        expect(requests[0].searchParams.get('session_id')).toBe('session-9');
    });
});
