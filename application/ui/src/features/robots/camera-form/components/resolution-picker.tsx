import { Item, Picker } from '@geti-ui/ui';

import { CameraFingerprint } from '../../../cameras/fingerprint';
import { CameraDriver } from '../provider';
import { RealsenseOutputType, useSupportedFormats } from './use-camera-data';

interface Resolution {
    width: number;
    height: number;
    fps: number[];
}

interface ResolutionPickerProps {
    driver: CameraDriver;
    fingerprint: CameraFingerprint | null | undefined;
    selectedResolution: { width: number | undefined; height: number | undefined } | undefined;
    onSelect: (resolution: Resolution) => void;
    outputType?: RealsenseOutputType;
}

export const ResolutionPicker = ({
    driver,
    fingerprint,
    selectedResolution,
    onSelect,
    outputType,
}: ResolutionPickerProps) => {
    const supportedResolutions = useSupportedFormats(driver, fingerprint, outputType);

    const selectedKey =
        selectedResolution?.width && selectedResolution?.height
            ? `${selectedResolution.width}_${selectedResolution.height}`
            : undefined;

    const isDisabled = !fingerprint;

    return (
        <Picker
            label='Resolution'
            width='100%'
            selectedKey={selectedKey}
            isDisabled={isDisabled}
            onSelectionChange={(key) => {
                const found = supportedResolutions.find(({ width, height }) => `${width}_${height}` === key);

                if (found) {
                    onSelect(found);
                }
            }}
        >
            {supportedResolutions.map(({ width, height }) => (
                <Item key={`${width}_${height}`}>{`${width} x ${height}`}</Item>
            ))}
        </Picker>
    );
};
