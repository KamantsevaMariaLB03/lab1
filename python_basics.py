"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput

#провалено
def count_vowels(data: TextInput) -> int:
    vowels = set('aeiou')
    return sum(1 for ch in data.value.lower() if ch in vowels)


def has_unique_characters(data: TextInput) -> bool:
    return len(data.value)==len(set(data.value))


def count_one_bits(data: PositiveIntegerInput) -> int:
    return bin(data.value).count('1')


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    n = data.value
    steps=0
    while n>=10:
        prod=1
        for digit in str(n):
            prod*=int(digit)
        n=prod
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    n = len(data.expected)
    if n==0:
        return 0.0
    total=sum((y_true - y_pred)**2 for y_true, y_pred in zip(data.expected, data.predicted))
    return total/n


def prime_factorization(data: PositiveIntegerInput) -> str:
    n = data.value
    factors = []
    d = 2
    while d * d <= n:
        count = 0
        while n % d == 0:
            count += 1
            n //= d
        if count > 0:
            factors.append(f"({d}**{count})" if count > 1 else f"({d})")
        d += 1
    if n > 1:
        factors.append(f"({n})")
    return "".join(factors)



def pyramid(data: PositiveIntegerInput) -> int | str:
    n = data.value
    k = 0
    while n > 0:
        k += 1
        n -= k * k
    return k if n == 0 else "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    s = str(data.value)
    length = len(s)
    mid = length // 2
    if length % 2 == 1:
        left = s[:mid]
        right = s[mid + 1:]
    else:
        left = s[:mid - 1]
        right = s[mid + 1:]
    return sum(int(d) for d in left) == sum(int(d) for d in right)
