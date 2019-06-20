
import sys
# path to the custom module
sys.path.append(r'C:\Users\Sameera\Documents\Github\Lets-Play-With-Pytorch')

from SkunkWork import Trainer as sw

if __name__ == "__main__":
    print('main')
    swd = sw.dSet()
    swd.compile()