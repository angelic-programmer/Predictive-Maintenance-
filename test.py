
class A:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    @staticmethod
    def product():
        return "Class A"
    
    def __str__(self):
        return f"Angelica(a={self.a}, b={self.b}, c={self.c})"
    def __repr__(self):
        return f"Angelica(a={self.a}, b={self.b}, c={self.c})"

a = A(2.5, 3.5, 4.5)
print(a)
print(A.product())

