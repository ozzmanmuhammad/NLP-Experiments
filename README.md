# NLP Experiments with Text Generation, Summarization & Sentiment Analysis

This repository contains details of multiple Natural Language Processing (NLP) experiments aimed at understanding different models available at Huggingface. All of these experiments were conducted using the Huggingface pipeline, a powerful object encapsulating various other pipelines. The pipeline abstraction serves as a wrapper around all the other available pipelines and is instantiated by choosing the task-specific pipelines. Visit the [Huggingface page](https://huggingface.co/transformers/main_classes/pipelines.html) for more details.
![NLP](https://github.com/ozzmanmuhammad/NLP-Experiments/assets/93766242/18bd3ce0-18c9-4244-9c0e-ad0a028ec96b)


## Tasks and Models

The following NLP tasks were explored:

1. **Text Generation** using GPT-Neo models:
   - GPT-Neo-2.7B (Not used due to memory constraints)
   - GPT-Neo-1.3B (Used for experiments)

2. **Text Summarization** using BART-CNN:
   - BART-CNN (Used for summarizing web articles)

3. **Sentiment Analysis** using roBERTa and NaiveBayes:
   - roBERTa (Used for sentiment analysis on English tweets)
   - NaiveBayes (Trained on nltk movie reviews corpus)

## Experiments and Results

### Text Generation - GPT-Neo

For text generation, I initially intended to use GPT-3, an attention-based model trained on a massive amount of data with a significant number of parameters. However, GPT-3 is only accessible through the OpenAI API and not publicly available. As an alternative, I used GPT-Neo, which is available on Huggingface and was trained on a substantial amount of data (2.7 Billion text datasets) though not as extensive as GPT-3.

Due to memory limitations on Google Colab, I encountered out-of-memory problems when using GPT-Neo-2.7B. As a solution, I switched to a smaller version, GPT-Neo-1.3B, which was still effective for text generation.

#### Model Inferences:

**Prompt Text**: "What is the meaning of life?"

**Output**: "What is the meaning of life? And is it to be considered as an activity? Life's Meaning is not limited to Activities Life is not an activity but the most important meaning of life is the meaning of life itself. But we always thought like that, we used the word 'determinism'. But the definition of determinism is 'ability of being a deterministic system'. If a person cannot be a determinist by his mind itself, then this person cannot exist."

**Prompt Text**: "To love is to suffer"

**Output**: "To love is to suffer. Love, I believe, is the root of suffering. It is the passion for the experience of being in love. Love is a feeling that is difficult to describe but is ultimately an experience of feeling it in a way that cannot be described. Love is like a fire; a consuming consuming, burning passion that burns down mountains, burns through walls, and burns out the sky. But the best thing in the world is to love."

### Text Summarization - BART-CNN

For text summarization, I utilized the "Extractive Summarization" approach using BART-CNN, the default model in the summarization pipeline. To gather data for summarization, I performed web scraping using BeautifulSoup to extract specific portions from web articles. To accommodate model constraints, I split the articles into 500-token chunks for inference.

#### Model Inferences:

Original Article Length: 649
Summary length: 69

**Article**: "Understanding-LSTMs"
**Output**: "Recurrent neural networks are networks with loops in them, allowing information to persist . These loops make them seem kind of mysterious, but they aren’t all that different than a normal neural network . LSTMs are a very special kind of recurrent neural network which works, for many tasks, much much better than the standard version . Long Short Term Memory networks are a special kind of RNN, capable of learning long-term dependencies . They work tremendously well on a large variety of problems, and are now widely used . The key to LSTMs is the cell state, the horizontal line running through the top of the diagram . The cell state is kind of like a conveyor belt, with only some minor linear interactions . LSTM has three gates, to protect and control the cell state . They are composed out of a sigmoid neural net layer and a pointwise multiplication operation . The first step is to decide what information we’re going to throw away from the state . In the example of our language model, we want to add the gender of the new subject to the state, so that the correct pronouns can be used . The previous steps already decided what to do, we need to actually do it . Not all LSTMs are the same as the above; almost every paper uses a slightly different version . The Gated Recurrent Unit, or GRU, combines the forget and input gates into a single 'update gate' The resulting model is simpler than standard LSTM models and growing increasingly popular . A common opinion among researchers is: 'Yes! There is a next step and it’s attention!' The idea is to let every step of an RNN pick information to look at from some larger collection of information . Grid LSTMs by Kalchbrenner, et al. (2015) seem extremely promising . The last few years have been an exciting time for recurrent neural networks."

### Text Sentiment Analysis - roBERTa and NaiveBayes

For sentiment analysis, I conducted two experiments. The first experiment involved using roBERTa, a model trained on ~58M tweets and fine-tuned for sentiment analysis with the TweetEval benchmark. The second experiment involved building a NaiveBayes model, trained on nltk's movie reviews corpus.

Both of these models were evaluated on 5000 tweets, gathered using the Twitter API and the Python tweepy library, regarding a specific topic (#AuratMarch2022). However, due to differences in the content of the tweets from the data on which the models were trained, the models performed poorly with incorrect classifications.

## Future Experiments

In the future, I plan to conduct the following experiments:

1. **Text Generation**:
   - Use GPT-2 for text generation

2. **Sentiment Analysis**:
   - Perform sentiment analysis on Yelp reviews
   - Utilize Abstractive Summarization with Pegasus

3. **Building Custom Model**:
   - Develop a custom model using research papers

For the source code and more details, please visit [my GitHub page](https://github.com/YourUsername/NLP-Experiments).
