import matplotlib.pyplot as plt


def draw_line(x, y_s, colors, labels, shapes, begins, lines_d, l_colors, step=1, x_name="STEP", y_name="LOSS"):
    fig = plt.figure(figsize=(6, 4), facecolor="white")
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel(x_name)
    ax1.set_ylabel(y_name)
    for idx, y in enumerate(y_s):
        x_ = x[begins[idx]:]
        y_ = y[begins[idx]:]
        ax1.scatter(x_[::step], y_[::step], color=colors[idx], marker=shapes[idx], edgecolors=colors[idx],  s=20, label=labels[idx])
        if lines_d[idx]:
            line_general = 20
            x_n = x_[::line_general]
            y_n = []
            sum_ = 0.
            sum_idx_num = 0
            for idx__, y__ in enumerate(y_):
                sum_idx_num += 1
                sum_ += y__
                if idx__ > 0 and idx__ % line_general == 0:
                    y_n.append(sum_ / line_general)
                    sum_ = 0.
                    sum_idx_num = 0
            if sum_ > 0.:
                y_n.append(sum_ / sum_idx_num)
            if idx == 0:
                ax1.plot(x_n[:29], y_n[:29], color="PaleGreen", linewidth=2.6, linestyle="-")
                ax1.plot(x_n[28:], y_n[28:], color="Green", linewidth=2.6, linestyle="-")
            else:
                ax1.plot(x_n, y_n, color=l_colors[idx], linewidth=2.6, linestyle="-")
    plt.vlines(542, 0, 2, colors="black", linestyles="dashed", linewidth=2)
    plt.annotate("warm up", xy=(300, 0.45), xytext=(250, 0.15), arrowprops=dict(facecolor="white", headlength=4,
                                                                               headwidth=13, width=4))
    plt.legend(loc="upper left")
    plt.grid(linestyle='-')
    plt.show()


if __name__ == "__main__":
    draw_line([1, 2, 3], [[3, 2, 1]], ["red"], ["A"])
