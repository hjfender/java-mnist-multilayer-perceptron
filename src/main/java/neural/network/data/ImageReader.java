package neural.network.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

public class ImageReader {

	/**
	 * Reads in the MNIST dataset.
	 * 
	 * Use absolute filepath (i.e. use '/' at front of parameter 'filepath').
	 * 
	 * @param filepath
	 * @return
	 * @throws IOException
	 */
	public static List<Image> read(final String filepath) throws IOException {
		List<Image> dataset = new ArrayList<Image>();
		
		URL resource = ImageReader.class.getResource(filepath);
		
		BufferedReader in = new BufferedReader(new InputStreamReader(resource.openStream()));

        String inputLine;
        
        while ((inputLine = in.readLine()) != null) {
        	String[] splitLine = inputLine.split(",");
        	Image img = new Image(Integer.parseInt(splitLine[0]));
        	for(int i = 1; i < splitLine.length; i++) {
        		img.setPx(i-1, Integer.parseInt(splitLine[i]));
        	}
        	dataset.add(img);
        }
            
        in.close();
        
		
		
		return dataset;
	}
}
