package iso.tool;

import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

public final class Main {
    public static void main(String[] args) throws Exception {
        if (args.length == 0 || "--gui".equalsIgnoreCase(args[0])) {
            UnifiedGui.launch();
            return;
        }
        Path root = Paths.get(args[0]);
        Path out = Paths.get(args.length > 1 ? args[1] : "iso-tool-output");
        Files.createDirectories(out.resolve("manifests"));
        DependencyDetector.write(out.resolve("manifests/dependencies.json"));
        List<Path> py = PythonParityScanner.scan(root);
        Files.write(out.resolve("manifests/python-modules.txt"), py.stream().map(Path::toString).sorted().collect(Collectors.toList()));
        System.out.println("ISO-Tool Java parity scan: " + py.size() + " Python modules");
    }
}
