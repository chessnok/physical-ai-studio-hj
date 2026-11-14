import { ComponentProps, KeyboardEvent, useRef, useState } from 'react';

import {
    ActionButton,
    Button,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    Loading,
    Switch,
    Text,
    TextField,
    toast,
    View,
} from '@geti-ui/ui';
import { useQueryClient } from '@tanstack/react-query';

import { $api, fetchClient } from '../../api/client';
import { getApiErrorMessage } from '../../api/errors';
import { SchemaStateSnapshot, SchemaStateSnapshotLabelUpdate } from '../../api/openapi-spec';
import { formatKeyCombo, keyboardEventToCombo } from '../hotkeys/key-combo';
import { snapshotListKey, useStateSnapshot } from './state-snapshot';
import { COMMENT_MAX_LENGTH } from './state-snapshot-label';

const RECENT_SNAPSHOTS = 8;
const ERROR_TOAST = { timeout: 10_000 };
const TEXT_INPUT_TYPES = new Set(['text', 'search', 'email', 'number', 'password', 'tel', 'url']);

/** Whether a keystroke into `target` edits text; a switch's checkbox input does not. */
const isTextEntry = (target: EventTarget): boolean =>
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement ||
    (target instanceof HTMLInputElement && TEXT_INPUT_TYPES.has(target.type)) ||
    (target instanceof HTMLElement && target.isContentEditable);

/** "20261114-031439-950" -> "03:14:39.950", the snapshot's local capture time ("-1" and so on kept). */
export const snapshotTime = (snapshot_id: string): string => {
    const match = /^\d{8}-(\d{2})(\d{2})(\d{2})-(\d{3})(-\d+)?$/.exec(snapshot_id);
    return match === null ? snapshot_id : `${match[1]}:${match[2]}:${match[3]}.${match[4]}${match[5] ?? ''}`;
};

/**
 * Keyboard handlers for an element inside which the capture hotkey (Space by default) always
 * captures, except while typing: a focused switch or button there is neither flipped nor pressed,
 * so that a label never changes by accident.
 */
const useCaptureKeys = (combo: string | undefined, capture: (() => void) | undefined) => {
    const ref = useRef<HTMLDivElement>(null);

    const isHotkey = (event: KeyboardEvent<HTMLDivElement>) =>
        combo !== undefined &&
        event.target instanceof Node &&
        // Its own element only: the recent list's popover is a portal, with handlers of its own.
        ref.current?.contains(event.target) === true &&
        !isTextEntry(event.target) &&
        keyboardEventToCombo(event.nativeEvent) === combo;
    const onKeyDownCapture = (event: KeyboardEvent<HTMLDivElement>) => {
        if (isHotkey(event)) {
            event.preventDefault();
            event.stopPropagation();
            if (!event.repeat) {
                capture?.();
            }
        }
    };
    // The browser activates a focused switch or button when Space is released.
    const onKeyUpCapture = (event: KeyboardEvent<HTMLDivElement>) => {
        if (isHotkey(event)) {
            event.preventDefault();
            event.stopPropagation();
        }
    };

    return { ref, onKeyDownCapture, onKeyUpCapture };
};

type TextFieldKeyHandler = NonNullable<ComponentProps<typeof TextField>['onKeyDown']>;

/** Enter (and Escape) leave the comment field, so that the capture hotkey works again. */
const leaveOnEnterOrEscape: TextFieldKeyHandler = (event) => {
    if (event.key === 'Enter' || event.key === 'Escape') {
        (event.target as HTMLElement).blur();
    } else {
        event.continuePropagation();
    }
};

/** One recent snapshot, with its label to change; one taken before labels existed has none yet. */
const RecentSnapshot = ({ project_id, snapshot }: { project_id: string; snapshot: SchemaStateSnapshot }) => {
    const queryClient = useQueryClient();
    const savedComment = snapshot.comment ?? '';
    const [draft, setDraft] = useState(savedComment);
    const [shown, setShown] = useState(savedComment);
    const isCancelled = useRef(false);
    // Adjusted while rendering: a comment saved elsewhere replaces the draft.
    if (shown !== savedComment) {
        setShown(savedComment);
        setDraft(savedComment);
    }
    const { mutate, isPending, variables } = $api.useMutation(
        'patch',
        '/api/projects/{project_id}/state_snapshots/{snapshot_id}',
        // Pending until the list shows the change, so that the switch does not flip back meanwhile.
        { meta: { awaits: [snapshotListKey(project_id)] } }
    );
    const time = snapshotTime(snapshot.snapshot_id);

    const update = (body: SchemaStateSnapshotLabelUpdate) => {
        mutate(
            { params: { path: { project_id, snapshot_id: snapshot.snapshot_id } }, body },
            {
                onError: (error) => {
                    const reason = getApiErrorMessage(error) ?? 'the backend did not answer.';
                    toast.negative(`Could not change the snapshot of ${time}: ${reason}`, ERROR_TOAST);
                    // Part of the change may have been saved: show what the backend has.
                    void queryClient.invalidateQueries({ queryKey: snapshotListKey(project_id) });
                },
            }
        );
    };
    const saveComment = () => {
        if (isCancelled.current) {
            isCancelled.current = false;
        } else if (draft.trim() !== savedComment) {
            update({ comment: draft });
        }
    };
    // Enter leaves the field, which saves it; Escape drops the edit and closes the list.
    const onCommentKeyDown: TextFieldKeyHandler = (event) => {
        if (event.key === 'Enter') {
            (event.target as HTMLElement).blur();
            return;
        }
        if (event.key === 'Escape') {
            isCancelled.current = true;
            setDraft(savedComment);
        }
        event.continuePropagation();
    };
    const defective = (isPending ? variables?.body?.defective : undefined) ?? snapshot.defective ?? null;

    return (
        <Flex direction='column' gap='size-50'>
            <Flex alignItems='center' justifyContent='space-between' gap='size-100'>
                <Text>{time}</Text>
                {defective === null ? (
                    <Flex alignItems='center' gap='size-100'>
                        <Text>Unlabelled</Text>
                        <ActionButton
                            onPress={() => update({ defective: false })}
                            aria-label={`Good, snapshot of ${time}`}
                        >
                            Good
                        </ActionButton>
                        <ActionButton
                            onPress={() => update({ defective: true })}
                            aria-label={`Defective, snapshot of ${time}`}
                        >
                            Defective
                        </ActionButton>
                    </Flex>
                ) : (
                    <Switch
                        isEmphasized
                        isSelected={defective}
                        onChange={(value) => update({ defective: value })}
                        aria-label={`Defective, snapshot of ${time}`}
                    >
                        Defective
                    </Switch>
                )}
            </Flex>
            <TextField
                aria-label={`Comment, snapshot of ${time}`}
                width='100%'
                value={draft}
                onChange={setDraft}
                onBlur={saveComment}
                onKeyDown={onCommentKeyDown}
                maxLength={COMMENT_MAX_LENGTH}
            />
        </Flex>
    );
};

const RecentSnapshotList = ({ project_id }: { project_id: string }) => {
    const live = useStateSnapshot();
    const keys = useCaptureKeys(live?.combo, live?.capture);
    const { data, isPending, error } = $api.useQuery('get', '/api/projects/{project_id}/state_snapshots', {
        params: { path: { project_id }, query: { limit: RECENT_SNAPSHOTS } },
    });

    const downloadCsv = () => {
        const url = fetchClient.PATH('/api/projects/{project_id}/state_snapshots/snapshots.csv', {
            params: { path: { project_id } },
        });
        window.open(url, '_blank', 'noopener,noreferrer');
    };

    let list;
    if (isPending) {
        list = <Loading mode='inline' />;
    } else if (error) {
        list = <Text>Could not list the snapshots: {getApiErrorMessage(error) ?? 'the backend did not answer.'}</Text>;
    } else if (data.length === 0) {
        list = <Text>No snapshots yet.</Text>;
    } else {
        list = data.map((snapshot) => (
            <RecentSnapshot key={snapshot.snapshot_id} project_id={project_id} snapshot={snapshot} />
        ));
    }

    return (
        // Space captures here too: after a switch here is clicked it keeps focus, and Space must not flip it back.
        <div ref={keys.ref} onKeyDownCapture={keys.onKeyDownCapture} onKeyUpCapture={keys.onKeyUpCapture}>
            <Flex direction='column' gap='size-200'>
                {list}
                <Button variant='secondary' onPress={downloadCsv}>
                    Download CSV of all snapshots
                </Button>
            </Flex>
        </div>
    );
};

/** The project's latest snapshots, to correct their labels, and the CSV of all of them. */
const RecentSnapshots = ({ project_id }: { project_id: string }) => {
    const isClosing = useRef(false);

    return (
        <DialogTrigger
            type='popover'
            onOpenChange={(isOpen) => {
                isClosing.current = !isOpen;
            }}
        >
            <ActionButton
                isQuiet
                // Closed, the list gives focus back to this button; let it go, so that Enter and Escape
                // reach the page's hotkeys (start, accept or discard an episode) instead of reopening it.
                onFocus={(event) => {
                    if (isClosing.current) {
                        isClosing.current = false;
                        (event.target as HTMLElement).blur();
                    }
                }}
            >
                Recent
            </ActionButton>
            <Dialog>
                <Heading>Recent snapshots</Heading>
                <Divider />
                <Content>
                    <RecentSnapshotList project_id={project_id} />
                </Content>
            </Dialog>
        </DialogTrigger>
    );
};

/**
 * The live session's snapshot controls: the label of the next snapshots (Defective, a comment),
 * which stays for a series and is remembered per project, a Capture button, and the recent snapshots.
 *
 * The capture hotkey (Space by default) captures with this label anywhere on the page except while
 * typing; within these controls and the recent list it also does on a switch or button, instead of
 * flipping or pressing it, so that a label never changes by accident. Enter or Escape leaves the
 * comment field without reaching the page's other hotkeys, and clicking a control here gives focus
 * back to the page. Nothing is shown unless the session has an environment.
 */
export const StateSnapshotControls = ({ layout = 'column' }: { layout?: 'column' | 'row' }) => {
    const snapshot = useStateSnapshot();
    const keys = useCaptureKeys(snapshot?.combo, snapshot?.capture);
    const controls = keys.ref;

    if (snapshot === null) {
        return null;
    }
    const { project_id, label, setLabel, capture, isCapturing, combo } = snapshot;

    // After a click on the switch or the button, focus goes back to the page, so that its hotkeys
    // (Enter, Escape) work at once instead of pressing that control again.
    const releaseFocus = () => {
        const focused = document.activeElement;
        if (focused instanceof HTMLElement && controls.current?.contains(focused)) {
            focused.blur();
        }
    };

    const defective = (
        <Switch
            isEmphasized
            isSelected={label.defective}
            onChange={(value) => {
                setLabel({ defective: value });
                releaseFocus();
            }}
        >
            Defective
        </Switch>
    );
    const comment = (
        <TextField
            label='Comment'
            width='100%'
            value={label.comment}
            onChange={(value) => setLabel({ comment: value })}
            onKeyDown={leaveOnEnterOrEscape}
            maxLength={COMMENT_MAX_LENGTH}
        />
    );
    const captureButton = (
        <Button
            variant='accent'
            onPress={() => {
                capture();
                releaseFocus();
            }}
            isPending={isCapturing}
        >
            Capture
        </Button>
    );
    const hint = `${formatKeyCombo(combo)} captures with this label, which stays for the next snapshots.`;

    return (
        <div ref={controls} onKeyDownCapture={keys.onKeyDownCapture} onKeyUpCapture={keys.onKeyUpCapture}>
            {layout === 'row' ? (
                <Flex direction='column' gap='size-50'>
                    <Flex alignItems='end' gap='size-200' wrap>
                        {defective}
                        <View flex minWidth='size-3000'>
                            {comment}
                        </View>
                        {captureButton}
                        <RecentSnapshots project_id={project_id} />
                    </Flex>
                    <Text>{hint}</Text>
                </Flex>
            ) : (
                <View backgroundColor='gray-200' padding='size-150' borderRadius='regular'>
                    <Flex direction='column' gap='size-100'>
                        <Flex alignItems='center' justifyContent='space-between' gap='size-100'>
                            <Heading level={4} margin={0}>
                                State snapshot
                            </Heading>
                            <RecentSnapshots project_id={project_id} />
                        </Flex>
                        <Flex alignItems='center' justifyContent='space-between' gap='size-100'>
                            {defective}
                            {captureButton}
                        </Flex>
                        {comment}
                        <Text>{hint}</Text>
                    </Flex>
                </View>
            )}
        </div>
    );
};
