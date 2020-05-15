package edu.campus02.iwi.spark.lv;

public class WinConfig {

	public static void setupEnv() {
		
		if(isWindowsOS()) {
			System.setProperty("hadoop.home.dir",
					System.getProperty("user.dir")+"\\hadoop\\");			
		}
		
	}
	
	private static boolean isWindowsOS() {
		
		return System.getProperty("os.name")
					.toLowerCase().contains("win");
	}
	
}