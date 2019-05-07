Download credits.csv from here. https://drive.google.com/drive/folders/1JnQXDCsGAb75I4PRRMDHUO0WxmXT-usv?usp=sharing

This is needed to make the code work

Requirements:
Need a vote count for the international movies
Genres for the international movies
Need year

About the Code:

Simple:

Make a weighted rating formula for the simple recommender system
Qualified applies the weighted rating formula and is then sorted
Drops genre from the metadata file
The function contains the formula and and then returns the movies that match the genre

Content-Based:

Takes the metadata file and checks if any column for each row is NaN. If it is, then drop the movie since it's useless
Make a column called description that includes tagline and overview of the moview
Use TfidVectorizer to analyze words in the description and make a matrix of it
This will get the cosine_similarity score
Use the cosine similarity score to get the amount of movie recommendations you want to get

Content-Based Using Description, taglines, keywords, cast, director, and genres:

Merge credits and keywords into one column
Choose top 3 actors that appear and make director's name be weighted twice to influence recommendations
Use Count Vectorizer to create matrix
Count out how many times the keywords appear in the movieid
Keywords that only appear once are useless and keywords will be lowercase and plural words will be the same as their singular form
Use the cosine similarity again to find recommendations for the movies again
Also add how popular and how highly rated the movie is to make better recommendations

Collaborative Filtering:


