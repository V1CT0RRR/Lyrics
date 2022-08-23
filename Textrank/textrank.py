from preprocessing import Preprocessing
from graph import TextrankGraph

class Textrank:
    def __init__(self, document, k=10, modified=True):
        """ Initialize a Textrank Object

        Args:
            document (str): Document to find keywords from.
            k (int, optional): Number of desired keywords. Defaults to 10.
            modified (bool, optional): Use modified post-processing. Defaults to True.
        """

        self.k = k
        self.modified = modified

        self.pipeline(document)


    def pipeline(self, document):
        """ Pipeline for textrank algorithm

        Args:
            document (str): Document to find keywords from.
        """

        preprocessed = Preprocessing(document)

        self.graph = TextrankGraph(preprocessed, k=self.k, modified=self.modified)

        self.candidates = self.graph.candidates


    def show_candidates(self):
        """ Display found keywords
        """        

        print()
        for phrase, score in self.candidates:
            print("{:<4f}: {}".format(score, phrase))

if __name__ == '__main__':
    text = "Discrete output feedback sliding mode control of second order systems - a moving switching line approach The sliding mode control systems (SMCS) for which the switching variable is designed independent of the initial conditions are known to be sensitive to parameter variations and extraneous disturbances during the reaching phase. For second order systems this drawback is eliminated by using the moving switching line technique where the switching line is initially designed to pass the initial conditions and is subsequently moved towards a predetermined switching line. In this paper, we make use of the above idea of moving switching line together with the reaching law approach to design a discrete output feedback sliding mode control. The main contributions of this work are such that we do not require to use system states as it makes use of only the output samples for designing the controller. and by using the moving switching line a low sensitivity system is obtained through shortening the reaching phase. Simulation results show that the fast output sampling feedback guarantees sliding motion similar to that obtained using state feedback"
    Textrank(document=text).show_candidates()