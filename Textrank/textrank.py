import argparse
import pandas as pd

from tqdm import tqdm

from preprocessing import Preprocessing
from graph import TextrankGraph

class Textrank:
    def __init__(self, k=10, modified=True):
        """ Initialize a Textrank Object

        Args:
            k (int, optional): Number of desired keywords. Defaults to 10.
            modified (bool, optional): Use modified post-processing. Defaults to True.
        """

        self.k = k
        self.modified = modified


    def pipeline(self, document):
        """ Pipeline for textrank algorithm

        Args:
            document (str): Document to find keywords from.
        """

        preprocessed = Preprocessing(document)

        self.graph = TextrankGraph(preprocessed, k=self.k, modified=self.modified)

        self.candidates = self.graph.candidates

        return ", ".join([candidate for candidate, _ in self.candidates])


    def show_candidates(self):
        """ Display found keywords
        """        

        print()
        for phrase, score in self.candidates:
            print("{:<4f}: {}".format(score, phrase))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-k", type=int, help="number of keyword candidates to extract. default to 10", default=10)
    argparser.add_argument("-path", type=str, help="path to the lyrics csv file, default to none", default="")
    argparser.add_argument("-document", type=str, help="document to extract keywords from. default to none", default="")
    args = argparser.parse_args()

    textrank = Textrank(k=args.k)
    if len(args.document) > 0:
        textrank.pipeline(document=args.document)
        textrank.show_candidates()
    elif len(args.path) > 0:
        try:
            df_lyrics = pd.read_csv(args.path, encoding= 'unicode_escape')
            # results = df_lyrics['lyrics'].apply(lambda x: Textrank(document=x, k=args.k).get_candidates_str())
            lyric_keywords = []
            for lyric in tqdm(df_lyrics['lyrics']):
                keywords = textrank.pipeline(document=lyric)
                lyric_keywords.append(keywords)
            df_lyrics['keywords'] = lyric_keywords
            df_lyrics.to_csv(args.path)
        except FileNotFoundError as e:
            print("File not found")
        except Exception as e:
            print("Error opening file")
    else:
        text = "Discrete output feedback sliding mode control of second order systems - a moving switching line approach The sliding mode control systems (SMCS) for which the switching variable is designed independent of the initial conditions are known to be sensitive to parameter variations and extraneous disturbances during the reaching phase. For second order systems this drawback is eliminated by using the moving switching line technique where the switching line is initially designed to pass the initial conditions and is subsequently moved towards a predetermined switching line. In this paper, we make use of the above idea of moving switching line together with the reaching law approach to design a discrete output feedback sliding mode control. The main contributions of this work are such that we do not require to use system states as it makes use of only the output samples for designing the controller. and by using the moving switching line a low sensitivity system is obtained through shortening the reaching phase. Simulation results show that the fast output sampling feedback guarantees sliding motion similar to that obtained using state feedback"
        textrank.pipeline(document=text)
        textrank.show_candidates()