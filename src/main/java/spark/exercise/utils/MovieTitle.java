package spark.exercise.utils;

import java.io.Serializable;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class MovieTitle implements Serializable {

	private static final long serialVersionUID = -963481970624643593L;

	private final static Pattern YEAR = Pattern.compile(" \\([0-9]{4}\\)$");

	public int id;
	public String title;
	public int year;
	public String[] genres;

	public MovieTitle(int id, String title, int year, String[] genres) {
		this.id = id;
		this.title = title;
		this.year = year;
		this.genres = genres;
	}

	public int getId() {
		return id;
	}

	public String getTitle() {
		return title;
	}

	public int getYear() {
		return year;
	}

	public String[] getGenres() {
		return genres;
	}

	public void setId(int id) {
		this.id = id;
	}

	public void setTitle(String title) {
		this.title = title;
	}

	public void setYear(int year) {
		this.year = year;
	}

	public void setGenres(String[] genres) {
		this.genres = genres;
	}

	public static MovieTitle parseColonsFormat(String raw) {
		String[] parts = raw.split("::");
		if (parts.length == 3) {
			int id = Integer.parseInt(parts[0]);
			String title = parts[1].replaceAll(YEAR.pattern(), "");
			Matcher m = YEAR.matcher(parts[1]);
			int year = -1;
			if (m.find()) {
				try {
					year = Integer.parseInt(m.group()
							.replaceAll("[()]", "").trim());
				} catch (NumberFormatException exc) {
				}
			}
			String[] genres = parts[2].split("\\|");
			return new MovieTitle(id, title, year, genres);
		}
		return null;
	}

	@Override
	public String toString() {
		return "Movie [id=" + id + ", title=" + title + ", year=" + year
				+ ", genres=" + Arrays.toString(genres) + "]";
	}
}
