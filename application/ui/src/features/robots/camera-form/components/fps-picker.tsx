import { Item, Picker } from '@geti-ui/ui';

import { CameraFingerprint } from '../../../cameras/fingerprint';
import { CameraDriver } from '../provider';
import { RealsenseOutputType, useSupportedFormats } from './use-camera-data';

interface FpsPickerProps {
    driver: CameraDriver;
    fingerprint: CameraFingerprint | null | undefined;
    width: number | undefined;
    height: number | undefined;
    selectedFps: number | undefined;
    onSelect: (fps: number) => void;
    outputType?: RealsenseOutputType;
}

export const FpsPicker = ({
    driver,
    fingerprint,
    width,
    height,
    selectedFps,
    onSelect,
    outputType,
}: FpsPickerProps) => {
    const supportedFormats = useSupportedFormats(driver, fingerprint, outputType);

    const resolution = supportedFormats.find((r) => r.width === width && r.height === height);
    const supportedFps = resolution?.fps ?? [];
    const isDisabled = !fingerprint;

    return (
        <Picker
            label='Frames per second (FPS)'
            width='100%'
            isDisabled={isDisabled}
            selectedKey={selectedFps !== undefined ? String(selectedFps) : undefined}
            onSelectionChange={(key) => {
                if (key !== null) {
                    onSelect(Number(key));
                }
            }}
        >
            {supportedFps.map((fps) => (
                <Item key={fps}>{`${fps}`}</Item>
            ))}
        </Picker>
    );
};
