import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;

import java.io.File;

public class Csv2Arff {

    public static void main(String[] args) throws Exception {
        if (args.length == 0 || args.length > 2) {
            System.err.println("Usage: java Csv2Arff <input.csv> [output.arff]");
            System.exit(1);
        }

        String inputPath = args[0];
        String outputPath = args.length == 2
                ? args[1]
                : inputPath.replaceFirst("\\.[cC][sS][vV]$", "") + ".arff";

        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(inputPath));
        Instances data = loader.getDataSet();

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(outputPath));
        saver.writeBatch();

        System.out.println("Saved ARFF to " + outputPath);
    }
}
