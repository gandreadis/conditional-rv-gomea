from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

# Unpruned
# fos = "[0]|[1]|[2]|[3]|[4]|[5]|[6]|[7]|[8]|[9]|[10]|[11]|[12]|[13]|[14]|[15]|[16]|[17]|[18]|[19]|[4^5]|[18^19]|[11^12]|[13^14]|[16^17]|[6^7]|[15^16^17]|[0^1]|[9^10]|[6^7^8]|[4^5^6^7^8]|[11^12^13^14]|[2^3]|[9^10^11^12^13^14]|[15^16^17^18^19]|[9^10^11^12^13^14^15^16^17^18^19]|[4^5^6^7^8^9^10^11^12^13^14^15^16^17^18^19]|[0^1^2^3]|[0^1^2^3^4^5^6^7^8^9^10^11^12^13^14^15^16^17^18^19]"

# Pruned
fos = "[2]|[15]|[3^4]|[5^6]|[11^12]|[13^14]|[16^17]|[18^19]|[16^17^18^19]|[15^16^17^18^19]|[7^8]|[0^1]|[2^3^4]|[9^10]|[7^8^9^10]|[0^1^2^3^4]|[11^12^13^14]|[5^6^7^8^9^10]|[5^6^7^8^9^10^11^12^13^14]|[0^1^2^3^4^5^6^7^8^9^10^11^12^13^14]|[0^1^2^3^4^5^6^7^8^9^10^11^12^13^14^15^16^17^18^19]"

fos_sets = [eval(s.replace("^", ",")) for s in fos.split("|")]
fos_sets = sorted(fos_sets, key=lambda x: len(x))

cmap = plt.colormaps.get_cmap('viridis')
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

max_val = 0
for ind, f in enumerate(fos_sets):
    for v in f:
        max_val = max(v, max_val)
        rect = Rectangle((v, ind),
                         1, 1,
                         color=cmap(ind / (len(fos_sets) - 1)))

        ax.add_patch(rect)

ax.set_xlim([0, max_val + 1])
ax.set_ylim([0, len(fos_sets)])
ax.set_xticks([])
ax.set_yticks([])

plt.savefig("fos.png", bbox_inches="tight")
