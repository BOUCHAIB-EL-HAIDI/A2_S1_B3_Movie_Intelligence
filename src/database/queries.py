from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")

db = client["movie_database"]

collection = db["movies"]


#count the number of document(movies)

count = collection.count_documents({})

print(f"movies count : {count}")

#find highly rated movies

movies = collection.find(

    {
        "vote_average": {"$gt": 8}

    }
).limit(10)

for movie in movies:
    print(movie["title"], movie["vote_average"])


#Find Action movies

movies = collection.find({
    "genres":"Action"
}).limit(10)

for movie in movies:
    print(movie["title"], movie["genres"])

#sort movies py popularity

movies = collection.find().sort("popularity", -1).limit(10)

for movie in movies :
    print(movie["title"] , movie["popularity"])


#aggregation by genre using 

pipline = [
     {"$unwind":"$genres"},
     {
         "$group":{
             "_id":"$genres",
             "movie_count":{"$sum": 1}     
         }     
    },
    {"$sort":{"movie_count":-1 }},
    {
        "$project":{
            "_id":0,
            "genre":"$_id",
            "movie_count":1
        }
    }
]

results = collection.aggregate(pipline)

for result in results:
    print(result)
