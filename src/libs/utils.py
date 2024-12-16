import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from base_libs import *

def check_folder_exist(*args, **kwargs):
    if len(args) != 0:
        for path in args:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

    if len(kwargs) != 0:
        for path in kwargs.values():
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

def delete_file_cronj(path_folder, stime, format_time='%Y-%m-%d_%H-%M_%S.jpg', ratio_delete=0.5):
	while True:
		time.sleep(stime)
		list_f = os.listdir(path_folder)
		list_f_sorted = sorted(list_f, key=lambda t: datetime.strptime(t, format_time))
		num_delete = round(len(list_f_sorted)/2*ratio_delete+1e-5)*2
		for i in range(num_delete):
			os.remove(os.path.join(path_folder,list_f_sorted[i]))

#---------------------------log---------------------------
logger.level("INFO", color="<light-green><dim>")
logger.level("DEBUG", color="<cyan><bold><italic>")
logger.level("WARNING", color="<yellow><bold><italic>")
logger.level("ERROR", color="<red><bold>")

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''

formatter = logging.Formatter(
    # fmt='\033[0;32m'+"%(asctime)s.%(msecs)03d | %(levelname)s    | %(name)s | %(message)s",
    fmt="%(asctime)s.%(msecs)03d | %(levelname)s    | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(formatter)
    
stdout_logger = logging.getLogger("stdout")
stdout_logger.setLevel(logging.INFO)
sl = StreamToLogger(stdout_logger, logging.INFO)
sys.stdout = sl

stderr_logger = logging.getLogger("stderr")
stderr_logger.setLevel(logging.ERROR)
sl = StreamToLogger(stderr_logger, logging.ERROR)
sys.stderr = sl
#////////////////////////////////////////////////////////////
