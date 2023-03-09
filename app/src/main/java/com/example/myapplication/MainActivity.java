package com.example.myapplication;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;

import org.apache.commons.lang3.tuple.Pair;
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

// read in csv
import com.opencsv.CSVReader;
import java.io.IOException;
import java.io.FileReader;

// lstm data per node
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    private Module mModule;
//    private List<String> mLocations = Arrays.asList("A","B","C","D","E","F","G","H","I","L","M","N","O","P");

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // create df, locations list, hours list
        List<List<Object>> df = new ArrayList<>(); // time, pillar, speed, angle
        List<String> locations = new ArrayList();
        List<Integer> hours = new ArrayList();

        try {
            File csvfile = new File(assetFilePath(getApplicationContext(), "short_filtered_21_50.csv"));
            CSVReader reader = new CSVReader(new FileReader(csvfile.getAbsolutePath()));
            String[] nextLine;
            Boolean isHeader = true;

            while ((nextLine = reader.readNext()) != null) {
                // nextLine[] is an array of values from the line
//                System.out.println(Arrays.toString(nextLine));
                if (isHeader) {
                    isHeader = false;
                    continue;
                }

                Integer hour = getHour(Integer.parseInt(nextLine[0]));
                if (!hours.contains(hour)) {
                    hours.add((hour));
                }

                List<Object> entry = new ArrayList<>();
                entry.add(nextLine[0]);
                entry.add(nextLine[1]);
                entry.add(nextLine[2]);
                entry.add(nextLine[3]);
                df.add(entry);

                if (!locations.contains(nextLine[1])) {
                    locations.add(nextLine[1]);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        // create lstm data per node map
        Map<String, Pair<Tensor, Tensor>> lstmDataPerNode = new HashMap<>();
        for (String location : locations) {
            float[] zerosArray = new float[1280];
            Arrays.fill(zerosArray, 0f);
            FloatBuffer zerosBuffer = Tensor.allocateFloatBuffer(zerosArray.length).put(zerosArray);
            Tensor zerosTensor = Tensor.fromBlob(zerosBuffer, new long[]{10, 128});
            Pair<Tensor, Tensor> zeroTensorsPair = Pair.of(zerosTensor, zerosTensor);
            lstmDataPerNode.put(location, zeroTensorsPair);
        }

//        System.out.println(df.toString());
        // load ptl model
        try {
            mModule = LiteModuleLoader.load(assetFilePath(getApplicationContext(), "ChangesToGraphMobile.ptl"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        // run model
        for (Integer hour : hours) {
            for (String location : locations) {
                List<List<Float>> dataSend = new ArrayList<>(); // list<speed, angle, time>
                for (List<Object> d : df) {
                    if (d.get(1).equals(location) && getHour(Integer.parseInt((String) d.get(0))) == hour) {
                        List<Float> dataSendEntry = new ArrayList<>();
                        dataSendEntry.add(Float.parseFloat((String) d.get(2)));
                        dataSendEntry.add(Float.parseFloat((String) d.get(3)));
                        dataSendEntry.add(Float.parseFloat((String) d.get(0)));
                        dataSend.add(dataSendEntry);
                    }
                }

                if (dataSend.size() < 1) {
                    continue;
                }

                float[][] data = new float[dataSend.size()][3];
                for (int i = 0; i < dataSend.size(); i++) {
                    List<Float> row = dataSend.get(i);
                    data[i][0] = row.get(0);
                    data[i][1] = row.get(1);
                    data[i][2] = row.get(2);
                }

                float[] flatArray = flatten(data);

                // Get the shape of the float array
                long[] shape = new long[] {data.length, data[0].length};

                // Create a Tensor from the flat array
                Tensor dataSendTo = Tensor.fromBlob(flatArray, shape);

                Pair<Tensor, Tensor> lstLSTM = lstmDataPerNode.get(location);
                IValue[] inputs = {IValue.from(dataSendTo), IValue.from(lstLSTM.getKey()), IValue.from(lstLSTM.getValue())};
                IValue outputTensor = mModule.forward(inputs);
                System.out.println(location + ": " + Arrays.toString(outputTensor.toTuple()[0].toTensor().getDataAsFloatArray()));
            }
        }
    }


    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }
        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    private static Integer getHour(Integer seconds) {
        Integer hour = (seconds / 3600 + 9) % 12;
        if (hour == 0) {
            hour = 12;
        }
        if (hour == 5) {
            hour = 4;
        }
        return hour;
    }

    private static float[] flatten(float[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        float[] flattenedData = new float[rows * cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flattenedData[i * cols + j] = data[i][j];
            }
        }
        return flattenedData;
    }
}
