package org.chimera.flashtool;

public final class FlashTool {
    public enum Transport { NONE, ADB, FASTBOOT, FASTBOOTD }

    public record DeviceInfo(String serial, String product, long ramBytes,
                             long storageBytes, boolean bootloaderUnlocked,
                             Transport transport) {}

    public record FlashPlan(String partition, String imagePath,
                            boolean dryRun, boolean verify) {}

    public static boolean validate(DeviceInfo device, FlashPlan plan) {
        if (device == null || plan == null || plan.partition().isBlank() || plan.imagePath().isBlank()) return false;
        if (device.transport() == Transport.NONE) return false;
        return plan.dryRun() || device.bootloaderUnlocked();
    }

    private FlashTool() {}
}
