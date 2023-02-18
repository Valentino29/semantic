import spacy

# Imported sklearn as found it more straightforward to use

from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_md")


def recommend_movie(description):
    # Load the movies from the file
    with open('movies.txt', 'r') as f:
        movies = f.readlines()

    # Create empty lists for titles and description and split information from text file into them
    titles = []
    descriptions = []
    for movie in movies:
        title, description = movie.split(':')
        titles.append(title.strip())
        descriptions.append(description.strip())

    # Find similarities between descriptions

    query_vec = nlp(description).vector
    similarities = [cosine_similarity(nlp(desc).vector.reshape(1,-1), query_vec.reshape(1,-1))[0][0] for desc in descriptions]

    # Find the index of the most similar movie
    index = similarities.index(max(similarities))

    # Return the title of the most similar movie with suggestion
    return "You might aswell like " + titles[index]


print(recommend_movie("The world at an end, a dying mother sends her young son on a quest to find the place that grants wishes."))