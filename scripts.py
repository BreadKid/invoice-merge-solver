import sys
import ast
from typing import List

Price = float  # 订单金额

'可参与抽奖的发票最小金额'
lucky_invoice_min_value = 100.0 


def _greedy(orders: List[Price], threshold: float = lucky_invoice_min_value) -> List[List[Price]]:
    """
    大规模回退方案：大订单做锚点 + 最小订单凑足（不保证全局最优）。
    输入/输出单位：元；内部按「分」做整数运算避免浮点误差。
    无法凑到 threshold 的剩余订单不输出。
    """
    from collections import deque

    k = round(threshold * 100)
    arr = sorted(round(x * 100) for x in orders)
    q = deque(arr)
    res: List[List[int]] = []
    while q:
        anchor = q.pop()
        if anchor >= k:
            res.append([anchor])
            continue
        group = [anchor]
        s = anchor
        while q and s < k:
            group.append(q.popleft())
            s += group[-1]
        if s >= k:
            res.append(group)
        else:
            q.extendleft(reversed(group))
            break
    return [[c / 100 for c in g] for g in res]


def _split_invoices(orders: List[Price],threshold: float = lucky_invoice_min_value) -> List[List[Price]]:
    """
    将订单组合成尽量多的发票，每张发票金额 >= max_value，每个订单最多用一次。

    精确解（订单数 <= 20）：
      1) 单个订单 >= max_value 时单独成票（必然最优）。
      2) 其余订单（均 < max_value）用状压 DP 求最大不相交分组数。
         只需枚举「最小有效组」——即金额 >= max_value 且去掉任一元素后即不足
         max_value 的组（等价：sum >= k 且 sum - min < k）。任何最优解都可改写为
         只用最小有效组：若某组含一个已达阈值的真子组，可用该子组替换并把多余
         订单并入其他发票，发票数不变。
      3) 无法凑到 max_value 的剩余订单不输出；全部无法成组时返回空列表。
    订单数 > 20 或候选组过多时回退到贪心近似解。
    内部以「分」为单位做整数运算，避免浮点误差。
    """
    if not orders:
        return []

    k = round(threshold * 100)
    cents = [round(x * 100) for x in orders]

    solo = [[c] for c in cents if c >= k]
    rest = [c for c in cents if c < k]
    invoices_cents: List[List[int]] = list(solo)

    if not rest:
        return [[c / 100 for c in g] for g in invoices_cents]

    n = len(rest)
    if n > 20:
        return _greedy(orders)

    size = 1 << n
    low_of = [0] * size
    sums = [0] * size
    minel = [0] * size

    def compute(prefix: List[int]):
        for m in range(1, size):
            lb = (m & -m).bit_length() - 1
            low_of[m] = lb
            prev = m ^ (1 << lb)
            sums[m] = sums[prev] + prefix[lb]
            minel[m] = prefix[lb] if prev == 0 else min(minel[prev], prefix[lb])

    # 第一遍：用原始顺序统计每个订单出现在多少个最小组里，用于重排位序
    compute(rest)
    membership = [0] * n
    first_minimal = []
    for m in range(1, size):
        if sums[m] >= k and sums[m] - minel[m] < k:
            first_minimal.append(m)
            mm = m
            while mm:
                b = (mm & -mm).bit_length() - 1
                membership[b] += 1
                mm &= mm - 1
        elif len(first_minimal) > 200000:
            break

    # 低位的订单尽量是“出现在最少组里”的，降低 DP 中最大分桶的工作量
    order = sorted(range(n), key=lambda i: membership[i])
    rest2 = [rest[i] for i in order]
    compute(rest2)

    groups_by_lowbit: List[List[int]] = [[] for _ in range(n)]
    for m in range(1, size):
        if sums[m] >= k and sums[m] - minel[m] < k:
            groups_by_lowbit[low_of[m]].append(m)

    if sum(len(g) for g in groups_by_lowbit) > 200000:
        return _greedy(orders)

    # DP：dp[mask] = 在 mask 子集内最多能成多少组（允许部分订单不用）
    dp = [0] * size
    choice = [-1] * size
    for m in range(1, size):
        lb = low_of[m]
        best = dp[m ^ (1 << lb)]  # 跳过最低位订单（留作剩余）
        best_g = -1
        for g in groups_by_lowbit[lb]:
            if (g & m) == g:
                cand = 1 + dp[m ^ g]
                if cand > best:
                    best = cand
                    best_g = g
        dp[m] = best
        choice[m] = best_g

    # 回溯选出分组
    full = size - 1
    selected: List[int] = []
    m = full
    while m:
        g = choice[m]
        if g < 0:
            m ^= 1 << low_of[m]
        else:
            selected.append(g)
            m ^= g

    used = 0
    for g in selected:
        used |= g
        invoices_cents.append([rest2[i] for i in range(n) if (g >> i) & 1])

    # 无法成组的剩余订单不输出；若没有任何有效发票则返回空
    if not invoices_cents:
        return []
    return [[c / 100 for c in g] for g in invoices_cents]

def _parse_orders(text: str) -> List[Price]:
    """解析命令行传入的订单列表，支持 JSON/Python 列表或逗号分隔。"""
    text = text.strip()
    if text.startswith("["):
        return [float(x) for x in ast.literal_eval(text)]
    return [float(x) for x in text.split(",") if x.strip() != ""]


def calculate(orders: List[Price], threshold: float = lucky_invoice_min_value):
    invoices = []
    if len(orders)<=20:
        invoices = _split_invoices(orders,threshold)
    else:
        invoices = _greedy(orders,threshold)
    for idx, inv in enumerate(invoices, 1):
        total = sum(inv)
        print(f"发票 {idx}: {inv}  共 {total:.2f} 元")
    print(f"共 {len(invoices)} 张发票，使用订单 {sum(len(i) for i in invoices)} / {len(orders)} 单")

if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv:
        orders = _parse_orders(argv[0])
        threshold = float(argv[1]) if len(argv) > 1 else lucky_invoice_min_value
    else:
        # 无参数时使用内置示例
        orders = [36.40, 29.70, 39.58, 27.57, 23.40, 27.70, 27.39, 28.80,
                  39.50, 42.30, 23.80, 41.52, 91.50, 33.38, 37.80, 39.90,
                  7.58, 40.79]
        threshold = 100.0
    calculate(orders, threshold)

