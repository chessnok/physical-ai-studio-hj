import { Flex, Item, NumberField, Picker, Text } from '@geti-ui/ui';

import { SchemaRealsenseCameraInput } from '../../../../api/openapi-spec';
import { CameraPicker } from '../components/camera-picker';
import { FpsPicker } from '../components/fps-picker';
import { NameField } from '../components/name-field';
import { ResolutionPicker } from '../components/resolution-picker';
import { RealsenseOutputType, useSupportedFormats } from '../components/use-camera-data';
import { useCameraFormFields } from '../components/use-camera-form-fields';
import { DriverFormSchema } from '../provider';

export const initialRealsenseState: DriverFormSchema<'realsense'> = {
    driver: 'realsense',
    payload: {
        depth_range_min: 0.3,
        depth_range_max: 3,
        output_type: 'color',
    },
};

export const validateRealsense = (formData: DriverFormSchema<'realsense'>): formData is SchemaRealsenseCameraInput => {
    const payload = formData.payload;
    return (
        !!formData.name &&
        !!formData.fingerprint &&
        !!payload?.width &&
        !!payload?.height &&
        !!payload?.fps &&
        (payload.depth_range_min ?? 0.3) < (payload.depth_range_max ?? 3)
    );
};

const OUTPUT_TYPES: { key: RealsenseOutputType; label: string }[] = [
    { key: 'color', label: 'Color' },
    { key: 'depth', label: 'Depth' },
    { key: 'both', label: 'Color + depth (aligned to color)' },
];

export const RealsenseFormFields = () => {
    const { formData, updateField, updatePayload } = useCameraFormFields('realsense');
    const outputType: RealsenseOutputType = formData.payload?.output_type ?? 'color';
    const hasDepth = outputType !== 'color';
    const supportedFormats = useSupportedFormats('realsense', formData.fingerprint, outputType);
    const { width, height, fps } = formData.payload ?? {};
    const isModeUnavailable =
        supportedFormats.length > 0 &&
        width !== undefined &&
        !supportedFormats.some(
            (format) => format.width === width && format.height === height && format.fps.includes(fps ?? -1)
        );

    return (
        <Flex gap='size-100' alignItems='end' direction='column'>
            <NameField value={formData.name ?? ''} onChange={(name) => updateField('name', name)} />
            <CameraPicker
                driver='realsense'
                selectedFingerprint={formData.fingerprint}
                onSelect={({ fingerprint, name }) => {
                    updateField('fingerprint', fingerprint);
                    updateField('hardware_name', name);
                }}
            />
            <Picker
                label='Output'
                width='100%'
                selectedKey={outputType}
                onSelectionChange={(key) => {
                    if (key !== null) {
                        updatePayload({ output_type: key as RealsenseOutputType });
                    }
                }}
            >
                {OUTPUT_TYPES.map(({ key, label }) => (
                    <Item key={key}>{label}</Item>
                ))}
            </Picker>
            <ResolutionPicker
                driver='realsense'
                fingerprint={formData.fingerprint}
                outputType={outputType}
                selectedResolution={{ width, height }}
                onSelect={(resolution) => {
                    const newFps = resolution.fps.find((f) => f === fps) ?? resolution.fps.at(-1) ?? 30;
                    updatePayload({ width: resolution.width, height: resolution.height, fps: newFps });
                }}
            />
            <FpsPicker
                driver='realsense'
                fingerprint={formData.fingerprint}
                outputType={outputType}
                width={width}
                height={height}
                selectedFps={fps}
                onSelect={(newFps) => updatePayload({ fps: newFps })}
            />
            {isModeUnavailable && (
                <Text width='100%'>
                    {`This camera does not offer ${width} x ${height} at ${fps} fps for this output on its current ` +
                        'USB connection. Pick another resolution or frame rate; 1280 x 720 at 30 fps needs USB 3.'}
                </Text>
            )}
            {hasDepth && (
                <Flex gap='size-100' width='100%'>
                    <NumberField
                        label='Nearest depth (m)'
                        width='50%'
                        minValue={0.1}
                        maxValue={5}
                        step={0.05}
                        value={formData.payload?.depth_range_min ?? 0.3}
                        onChange={(value) => updatePayload({ depth_range_min: value })}
                    />
                    <NumberField
                        label='Farthest depth (m)'
                        width='50%'
                        minValue={1}
                        maxValue={10}
                        step={0.1}
                        value={formData.payload?.depth_range_max ?? 3}
                        onChange={(value) => updatePayload({ depth_range_max: value })}
                    />
                </Flex>
            )}
            {hasDepth && (
                <Text width='100%'>
                    Depth is recorded in millimetres as a camera of its own, named after this one with
                    &quot;_depth&quot; appended, aligned to the color image. Depth outside this range is recorded as 0
                    (no data).
                </Text>
            )}
        </Flex>
    );
};
