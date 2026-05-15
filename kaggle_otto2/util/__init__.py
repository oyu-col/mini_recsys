from kaggle_otto2.util.cv_util import CvUtil
from kaggle_otto2.util.evaluate_util import EvaluateUtil
from kaggle_otto2.util.file_util import FileUtil
from kaggle_otto2.util.global_util import GlobalUtil
from kaggle_otto2.util.time_util import TimeUtil


def __getattr__(name):
    if name == "PlotUtil":
        from kaggle_otto2.util.plot_util import PlotUtil

        return PlotUtil
    if name == "SearchUtil":
        from kaggle_otto2.util.search_util import SearchUtil

        return SearchUtil
    raise AttributeError(name)
