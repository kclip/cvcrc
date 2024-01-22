

function plot_curves_with_conf_intervals(x, m_avg, m_std, color_str, line_str, MainDisplayName)%, MinorDisplayName)
    plot_curve_with_conf_interval(x, m_avg.', (m_avg - m_std).', (m_avg + m_std).' , color_str , line_str, MainDisplayName);%,' ',MinorDisplayName{mm}]);
end

function plot_curve_with_conf_interval(x, y_avg, y_min, y_max, color_str, line_str, DisplayName)
    shade_area( x,         y_min, y_max , color_str); hold on;
    semilogx(   x,         y_avg        , [color_str,line_str]  ,'LineWidth',1.5,'DisplayName',DisplayName);
end

function shade_area(x,curve1,curve2,color_str)
    x2 = [x, fliplr(x)];
    inBetween = [curve1, fliplr(curve2)];
    fill(x2, inBetween, color_str,'FaceAlpha',0.1,'LineStyle','none','HandleVisibility','off');
end