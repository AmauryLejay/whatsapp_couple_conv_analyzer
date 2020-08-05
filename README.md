![Whatsapp](https://github.com/AmauryLejay/whatsapp_couple_conv_analyzer/blob/master/image.png?raw=true)

### Whatsapp couple conversation analyzer

#### Why? 
Short weekend project aimed at buidling a [python class](https://github.com/AmauryLejay/whatsapp_couple_conv_analyzer/blob/master/whatsapp_couple_conversation_analyzer.py) to draw quick and meaninfull statistics about my partner and I whatsapp conversation. 
Current solutions offer even more basic analysis, where metrics measured are more directed toward group of friends conversations.

#### How to use it?
- Clone the project

```{python}
from whatsapp_couple_conversation_analyzer import whatsapp_analyzer
whatsapp = whatsapp_analyzer("_chat.txt",language = 'french',top_x_common_words = 30) 
df = whatsapp.analyse(specific_preprocessing = True)
```
- or follow the [jupyter notebook tutorial](https://github.com/AmauryLejay/whatsapp_couple_conv_analyzer/blob/master/tutorial.ipynb)
- Have fun! 

#### Limitations
- Primarily focused on conversations for couples at the moment.
- French language supported by default, possible to switch to english, italian or any other nltk supported language by changing the language setting (cf tutorial)

#### Would be nice to have in the future...
- a nice Dash visualization
- Tests (always)
- sentiment analysis (I could not find a free and good working french sentiment analyser, the ones I tested didn't give good results)
- favourit emojies analysis
- support for larger group conversation - not difficult but a bit more work and not in my interest at the moment.

Make good use of it if you do :wink:
