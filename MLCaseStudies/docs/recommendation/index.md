# Recommendation & Personalization

Six case studies on deciding what to show a user. The recommendation system is the canonical case: one problem that forces candidate generation, ranking, multi task objectives, cold start, position bias, and A/B testing. The others reuse the same skeleton. Feed ranking adds the engagement trap, conversational recommendation adds dialogue and constraints, graph recommendation adds candidate generation over a social graph, autocomplete adds query completion under tight latency, and notifications add bandits and a fatigue budget.

The shared intellectual core across all six is the feedback loop: the user only acts on what you showed, so your next training set is biased by your current model.

- [Recommendation System](01-recommendation-system.md)
- [News Feed / Personalized Ranking](02-news-feed-personalized-ranking.md)
- [Conversational Recommender](15-conversational-recommender.md)
- [People You May Know / Graph Recommendation](16-people-you-may-know-graph-recommendation.md)
- [Autocomplete / Typeahead Personalization](18-autocomplete-typeahead-personalization.md)
- [Notification Optimization / Bandit System](19-notification-optimization-bandit-system.md)
