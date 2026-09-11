package org.chimera.flashtool;

import java.nio.file.Files;
import java.nio.file.Path;

public final class FlashTool {
    public enum Transport { NONE, ADB, FASTBOOT, FASTBOOTD }
    public enum Slot { UNKNOWN, A, B, NON_AB }
    public enum ImageKind { UNKNOWN, ANDROID_BOOT, INIT_BOOT, VENDOR_BOOT, DTBO, SPARSE, VBMETA, SUPER, OTA_PAYLOAD, OTA_ZIP }

    public record DeviceInfo(String serial, String product, long ramBytes,
                             long storageBytes, boolean bootloaderUnlocked,
                             Transport transport, Slot slot,
                             boolean dynamicPartitions, boolean fastbootd) {
        public DeviceInfo(String serial, String product, long ramBytes,
                          long storageBytes, boolean bootloaderUnlocked,
                          Transport transport) {
            this(serial, product, ramBytes, storageBytes, bootloaderUnlocked,
                 transport, Slot.UNKNOWN, false, false);
        }
    }

    public record FlashPlan(String partition, String imagePath,
                            boolean dryRun, boolean verify,
                            String expectedSha256, Slot targetSlot) {
        public FlashPlan(String partition, String imagePath, boolean dryRun, boolean verify) {
            this(partition, imagePath, dryRun, verify, "", Slot.UNKNOWN);
        }
    }

    public record Preflight(boolean ok, String code, String reason,
                            boolean requiresConfirmation, long imageSize) {}

    public static boolean validate(DeviceInfo device, FlashPlan plan) {
        if (device == null || plan == null || plan.partition().isBlank() || plan.imagePath().isBlank()) return false;
        if (device.transport() == Transport.NONE) return false;
        if (!plan.dryRun() && !device.bootloaderUnlocked()) return false;
        if (plan.targetSlot() != Slot.UNKNOWN && plan.targetSlot() != Slot.NON_AB && plan.targetSlot() != device.slot()) return false;
        return true;
    }

    public static Preflight preflight(DeviceInfo device, FlashPlan plan, long partitionSize) {
        if (device == null || plan == null) return new Preflight(false, "invalid-input", "device and plan are required", true, 0);
        if (device.transport() == Transport.NONE) return new Preflight(false, "no-transport", "no supported transport", true, 0);
        if (!plan.dryRun() && !device.bootloaderUnlocked()) return new Preflight(false, "locked-write", "authorized unlocked state is required", true, 0);
        long imageSize = 0;
        try { imageSize = Files.size(Path.of(plan.imagePath())); } catch (Exception ignored) {}
        if (partitionSize > 0 && imageSize > partitionSize) return new Preflight(false, "image-too-large", "image exceeds target partition", true, imageSize);
        return new Preflight(true, "ok", "preflight passed; explicit confirmation remains required", true, imageSize);
    }

    private FlashTool() {}
}
