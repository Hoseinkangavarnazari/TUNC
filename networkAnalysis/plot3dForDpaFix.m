function plot3dForDpaFix()

    runs = 100000;
    subset_sizes = 9;
    hops = 8;
    % 初始化矩阵
    success_ratio_matrix = zeros(subset_sizes, hops);

    % 读取数据并计算成功率
    for hop = 1:hops
        for keys = 1:subset_sizes
            filename = sprintf('./dpa_fix/results_%druns_%dkeys_%dhops.csv', runs, keys, hop);
            data = readtable(filename);
            success_count = sum(strcmp(data.is_success, 'True'));
            success_ratios = success_count / runs;

            success_ratio_matrix(keys, hop) = success_ratios;
        end
    end

    % 绘制 3D 条形图
    bar3(success_ratio_matrix);

    % 设置坐标轴和标签
    ax = gca;
    set(ax, 'ZScale', 'log');
    xlabel('Hops');
    ylabel('Key Subset Size');
    zlabel('Logarithmic Success Ratio');
    title(sprintf('%d runs / key subset size / hop', runs));
    view(69, 23); % azimuth, elevation

    % 调整 Z 轴数据以适应对数刻度
    zlim = 10^(-10);
    h = get(gca, 'Children');
    for i = 1:length(h)
        zdata = get(h(i), 'ZData');
        zdata(zdata < zlim) = zlim;
        set(h(i), 'ZData', zdata);
    end

    % 设置图形尺寸并保存
    fig = gcf;
    fig.Position = [0, 0, 800, 600];
    print(fig, '-dpng', sprintf('./dpa_fix/plot_3d_%druns.png', runs), '-r300');
end
