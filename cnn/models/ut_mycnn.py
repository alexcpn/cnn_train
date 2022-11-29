import unittest
from model import mycnn

class TestMyCNN(unittest.TestCase):

    def test_output(self):
        model = mycnn.MyCNN()
        out= model.get_output(432,320,1,3,3)
        print(f"Output is width {out[0]} height {out[1]}, depth {out[2]}")
        self.assertEqual(out, (430,318,3))


if __name__ == '__main__':
    unittest.main()
