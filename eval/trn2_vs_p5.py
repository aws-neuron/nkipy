# %%
import matplotlib.pyplot as plt


plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

# Data
p5_ttft = [0.34, 0.68
#, 1.36
]
p5_tput = [4.8, 7.9
#, 14
]

trn2_ttft = [
  # 0.image.png28, 
0.42, 0.7]
trn2_tput = [
  # 1.819, 
3.594, 6.853]

# Scatter plot
plt.scatter(p5_ttft, p5_tput, label='p5', s=100, alpha=0.7)
plt.plot(p5_ttft, p5_tput, linestyle='--', alpha=0.5)
plt.scatter(trn2_ttft, trn2_tput, label='trn2', s=100, alpha=0.7)
plt.plot(trn2_ttft, trn2_tput, linestyle='--', alpha=0.5)

plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xlabel('Prefill TTFT (s)')
plt.ylabel('Decode tput (k tokens/s)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# %%