package spark.exercise.utils;

import java.io.Serializable;

public class MovieRating implements Serializable {
	
	private static final long serialVersionUID = 9022558062331400974L;

	private int userId;
	private int movieId;
	private double rating;
	private long timestamp;

	public MovieRating() {
	}

	public MovieRating(int userId, int movieId, double rating, long timestamp) {
		this.userId = userId;
		this.movieId = movieId;
		this.rating = rating;
		this.timestamp = timestamp;
	}

	public int getUserId() {
		return userId;
	}

	public int getMovieId() {
		return movieId;
	}

	public double getRating() {
		return rating;
	}

	public long getTimestamp() {
		return timestamp;
	}

	public void setUserId(int userId) {
		this.userId = userId;
	}

	public void setMovieId(int movieId) {
		this.movieId = movieId;
	}

	public void setRating(double rating) {
		this.rating = rating;
	}

	public void setTimestamp(long timestamp) {
		this.timestamp = timestamp;
	}

	public static MovieRating parseRating(String str) {
		String[] fields = str.split("::");
		if (fields.length != 4) {
			throw new IllegalArgumentException("Each line must contain 4 fields");
		}
		int userId = Integer.parseInt(fields[0]);
		int movieId = Integer.parseInt(fields[1]);
		double rating = Double.parseDouble(fields[2]);
		long timestamp = Long.parseLong(fields[3]);
		return new MovieRating(userId, movieId, rating, timestamp);
	}
}
