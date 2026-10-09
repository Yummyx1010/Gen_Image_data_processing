args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L)
run_dir <- args[1]; score_dir <- args[2]; out_dir <- args[3]
models <- c('A', 'F', 'B')
seeds <- c(42L, 123L, 2026L)
generators <- c('glide', 'Midjourney', 'wukong')
colors <- c(A = '#0072B2', F = '#D55E00', B = '#009E73')
model_names <- c(A = 'A: spatial only', F = 'F: frequency only', B = 'B: spatial + frequency')
read_table <- function(name) read.csv(file.path(score_dir, name), check.names = FALSE)
per_gen <- read_table('per_generator_metrics.csv')
stability <- read_table('per_seed_stability.csv')
stopifnot(nrow(per_gen) == 27L, nrow(stability) == 9L)

base_style <- function() {
  par(family = 'sans', las = 1, bty = 'l', tcl = -0.25, mgp = c(2.4, 0.7, 0),
      col.axis = '#333333', col.lab = '#222222', fg = '#333333', cex = 0.95)
}
render <- function(name, width, height, draw) {
  png(file.path(out_dir, paste0(name, '.png')), width = width, height = height,
      units = 'in', res = 240, type = 'quartz', bg = 'white')
  base_style(); draw(); invisible(dev.off())
}
whisker <- function(x, y, spread, color) {
  arrows(x, y - spread, x, y + spread, angle = 90, code = 3, length = 0.045,
         col = color, lwd = 1.5)
}

render('01_AUROC_by_generator', 9.4, 5.9, function() {
  par(mar = c(5.4, 4.5, 4.9, 1.5))
  plot(NA, xlim = c(0.5, 3.5), ylim = c(0.5, 1.015), xaxt = 'n', yaxt = 'n',
       xlab = '', ylab = 'Test AUROC (higher is better)')
  abline(h = seq(0.5, 1, 0.1), col = '#E8E8E8', lwd = 0.8)
  axis(1, at = 1:3, labels = generators, cex.axis = 1.05)
  axis(2, at = seq(0.5, 1, 0.1), labels = sprintf('%.1f', seq(0.5, 1, 0.1)))
  for (m in seq_along(models)) {
    model <- models[m]
    for (g in seq_along(generators)) {
      rows <- per_gen[per_gen$model == model & per_gen$generator == generators[g], ]
      rows <- rows[match(seeds, rows$seed), ]
      x <- g + (m - 2) * 0.24
      avg <- mean(rows$auroc); spread <- sd(rows$auroc)
      whisker(x, avg, spread, colors[model])
      points(x + c(-0.035, 0, 0.035), rows$auroc, pch = 16, cex = 0.9,
             col = adjustcolor(colors[model], alpha.f = 0.50))
      points(x, avg, pch = 18, cex = 1.55, col = colors[model])
      text(x, avg + spread + 0.013, sprintf('%.4f', avg), cex = 0.83,
           col = colors[model])
    }
  }
  title('Detection performance on unseen generators', line = 3.1, cex.main = 1.18)
  legend('top', inset = c(0, -0.16), xpd = NA, legend = model_names[models],
         col = colors[models], pch = 18, horiz = TRUE, bty = 'n', cex = 0.90)
  mtext('Dots: individual seeds. Diamonds and whiskers: mean +/- SD across 3 seeds.',
        side = 1, line = 3.0, cex = 0.80)
  mtext('Full test: 2,500 fake images per generator + the same 1,166 real images.',
        side = 1, line = 4.1, cex = 0.80)
})

render('02_mean_and_consistency', 10.6, 5.5, function() {
  par(mfrow = c(1, 2), oma = c(3.6, 0, 2.0, 0), mar = c(3.4, 4.7, 3.5, 1.0))
  for (metric in c('macro_auroc', 'cross_generator_sd')) {
    is_mean <- metric == 'macro_auroc'
    limits <- if (is_mean) c(0.5, 1.015) else c(0, 0.12)
    ticks <- if (is_mean) seq(0.5, 1.0, 0.1) else seq(0, 0.12, 0.02)
    plot(NA, xlim = c(0.5, 3.5), ylim = limits, xaxt = 'n', yaxt = 'n', xlab = '',
         ylab = if (is_mean) 'Macro AUROC (higher is better)' else 'Across-generator SD (lower is better)')
    abline(h = ticks, col = '#E8E8E8', lwd = 0.8)
    axis(1, at = 1:3, labels = models, cex.axis = 1.10)
    axis(2, at = ticks, labels = if (is_mean) sprintf('%.1f', ticks) else sprintf('%.2f', ticks))
    for (m in seq_along(models)) {
      model <- models[m]
      rows <- stability[stability$model == model, ]
      rows <- rows[match(seeds, rows$seed), ]
      values <- rows[[metric]]
      avg <- mean(values); spread <- sd(values)
      whisker(m, avg, spread, colors[model])
      points(m + c(-0.075, 0, 0.075), values, pch = 16, cex = 1.0,
             col = adjustcolor(colors[model], alpha.f = 0.50))
      points(m, avg, pch = 18, cex = 1.65, col = colors[model])
      text(m, max(values, avg + spread) + if (is_mean) 0.019 else 0.005,
           sprintf('%.4f', avg), col = colors[model], cex = 0.96)
    }
    title(if (is_mean) 'Average detection performance' else 'Cross-generator consistency',
          line = 2.1, cex.main = 1.05)
    mtext(if (is_mean) 'Each seed: mean AUROC across 3 generators' else 'Each seed: generator SD of AUROC (ddof = 0)',
          side = 3, line = 0.7, cex = 0.78)
  }
  mtext('Original full test: A, F and B; seeds 42, 123 and 2026', outer = TRUE,
        side = 3, line = 0.6, font = 2, cex = 1.0)
  mtext('Dots: individual seeds. Diamonds and whiskers: mean +/- SD across seeds (not confidence intervals).',
        outer = TRUE, side = 1, line = 1.0, cex = 0.76)
  mtext('A: spatial only     F: frequency only     B: spatial + frequency',
        outer = TRUE, side = 1, line = 2.2, cex = 0.83)
})

roc_curve <- function(labels, scores) {
  stopifnot(all(labels %in% c(0L, 1L)), all(is.finite(scores)))
  ord <- order(scores, decreasing = TRUE)
  scores <- scores[ord]; labels <- labels[ord]
  ends <- c(which(diff(scores) != 0), length(scores))
  positives <- sum(labels); negatives <- length(labels) - positives
  stopifnot(positives == 2500L, negatives == 1166L)
  x <- c(0, cumsum(1 - labels)[ends] / negatives)
  y <- c(0, cumsum(labels)[ends] / positives)
  value <- sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)
  list(x = x, y = y, auroc = value)
}
curves <- list(); checks <- list()
for (model in models) {
  for (seed in seeds) {
    path <- file.path(run_dir, sprintf('%s_seed_%s_predictions.csv', model, seed))
    rows <- read.csv(path, check.names = FALSE, colClasses = c(model = 'character'))
    stopifnot(nrow(rows) == 8666L, length(unique(rows$sample_id)) == 8666L,
              all(rows$model == model), all(rows$seed == seed))
    for (generator in generators) {
      group <- rows[rows$generator %in% c(generator, 'Nature'), ]
      curve <- roc_curve(group$true_label, group$pred_probability)
      expected <- per_gen[per_gen$model == model & per_gen$seed == seed &
                          per_gen$generator == generator, 'auroc']
      stopifnot(length(expected) == 1L, abs(curve$auroc - expected) < 1e-12)
      curves[[paste(model, seed, generator, sep = '_')]] <- curve
      checks[[length(checks) + 1L]] <- data.frame(model = model, seed = seed,
          generator = generator, n_images = nrow(group), calculated_auroc = curve$auroc,
          recorded_auroc = expected)
    }
  }
}
write.csv(do.call(rbind, checks), file.path(out_dir, 'roc_checks.csv'), row.names = FALSE)

render('03_ROC_by_generator', 11.8, 4.9, function() {
  par(mfrow = c(1, 3), oma = c(3.7, 0, 2.4, 0), mar = c(3.7, 4.0, 3.0, 0.9))
  grid <- seq(0, 1, length.out = 1001L)
  for (generator in generators) {
    plot(NA, xlim = c(0, 1), ylim = c(0, 1), xaxs = 'i', yaxs = 'i',
         xlab = 'False positive rate', ylab = 'True positive rate',
         main = generator, cex.main = 1.08, cex.lab = 0.92)
    abline(h = seq(0, 1, 0.2), v = seq(0, 1, 0.2), col = '#EFEFEF', lwd = 0.7)
    abline(0, 1, lty = 2, col = '#999999', lwd = 1.0)
    legend_labels <- character()
    for (model in models) {
      interpolated <- list(); aucs <- numeric()
      for (seed in seeds) {
        curve <- curves[[paste(model, seed, generator, sep = '_')]]
        lines(curve$x, curve$y, col = adjustcolor(colors[model], alpha.f = 0.25), lwd = 0.85)
        interpolated[[length(interpolated) + 1L]] <- approx(curve$x, curve$y,
             xout = grid, ties = max, rule = 2)$y
        aucs <- c(aucs, curve$auroc)
      }
      average_curve <- rowMeans(do.call(cbind, interpolated))
      lines(grid, average_curve, col = colors[model], lwd = 2.4)
      legend_labels <- c(legend_labels, sprintf('%s: mean AUROC %.4f', model, mean(aucs)))
    }
    legend('bottomright', inset = 0.035, legend = legend_labels, col = colors[models],
           lwd = 2.4, bty = 'n', cex = 0.75)
    mtext('2,500 fake + 1,166 real images', side = 3, line = 0.2, cex = 0.70)
  }
  mtext('ROC curves on unseen generators', outer = TRUE, side = 3,
        line = 0.9, font = 2, cex = 1.10)
  mtext('Thin curves: individual seeds. Thick curves: pointwise mean ROC. Legend: mean of per-seed AUROCs.',
        outer = TRUE, side = 1, line = 1.2, cex = 0.77)
  mtext('Original full test; seeds 42, 123 and 2026. No image filtering.',
        outer = TRUE, side = 1, line = 2.4, cex = 0.82)
})
cat('Completed three figures in PNG.\n')
cat('Verified 27 ROC AUROCs against the existing formal scores.\n')
cat(R.version.string, '\n')
