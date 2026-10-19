args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
score_dir <- args[1]
out_dir <- args[2]
models <- c('A', 'F', 'B', 'B2')
seeds <- c(42L, 123L, 2026L)
generators <- c('glide', 'Midjourney', 'wukong')
colors <- c(A = '#0072B2', F = '#D55E00', B = '#009E73', B2 = '#CC79A7')
model_names <- c(A = 'A: spatial only', F = 'F: frequency only',
                 B = 'B: trainable frequency fusion', B2 = 'B2: frozen frequency fusion')
read_table <- function(name) read.csv(file.path(score_dir, name), check.names = FALSE)
per_gen <- read_table('per_generator_metrics.csv')
stability <- read_table('per_seed_stability.csv')
generator_summary <- read_table('generator_summary.csv')
model_summary <- read_table('model_summary.csv')
stopifnot(nrow(per_gen) == 36L, nrow(stability) == 12L,
          nrow(generator_summary) == 12L, nrow(model_summary) == 4L,
          all(per_gen$n_real == 1166L), all(per_gen$n_fake == 2500L))

close_enough <- function(a, b) isTRUE(all.equal(as.numeric(a), as.numeric(b), tolerance = 1e-12))
for (model in models) {
  for (generator in generators) {
    rows <- per_gen[per_gen$model == model & per_gen$generator == generator, ]
    reference <- generator_summary[generator_summary$model == model &
                                   generator_summary$generator == generator, ]
    stopifnot(setequal(rows$seed, seeds), nrow(rows) == 3L, nrow(reference) == 1L,
              close_enough(mean(rows$auroc), reference$auroc_mean),
              close_enough(sd(rows$auroc), reference$auroc_seed_sd))
  }
  rows <- stability[stability$model == model, ]
  reference <- model_summary[model_summary$model == model, ]
  stopifnot(setequal(rows$seed, seeds), nrow(rows) == 3L, nrow(reference) == 1L)
  for (metric in c('macro_auroc', 'cross_generator_sd')) {
    stopifnot(close_enough(mean(rows[[metric]]), reference[[paste0(metric, '_mean')]]),
              close_enough(sd(rows[[metric]]), reference[[paste0(metric, '_seed_sd')]]))
  }
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
render <- function(name, width, height, draw) {
  path <- file.path(out_dir, paste0(name, '.png'))
  if (file.exists(path)) stop('Output already exists: ', path)
  png(path, width = width, height = height, units = 'in', res = 300,
      type = 'quartz', bg = 'white')
  par(family = 'sans', las = 1, bty = 'l', tcl = -0.25, mgp = c(2.4, 0.7, 0),
      col.axis = '#333333', col.lab = '#222222', fg = '#333333', cex = 0.95)
  draw()
  invisible(dev.off())
}
whisker <- function(x, y, spread, color) {
  arrows(x, y - spread, x, y + spread, angle = 90, code = 3,
         length = 0.045, col = color, lwd = 1.5)
}

render('01_AUROC_by_generator_B2', 10.8, 6.3, function() {
  par(mar = c(5.4, 4.5, 6.0, 1.3))
  plot(NA, xlim = c(0.45, 3.55), ylim = c(0.5, 1.04), xaxt = 'n', yaxt = 'n',
       xlab = '', ylab = 'Test AUROC (higher is better)')
  abline(h = seq(0.5, 1.0, 0.1), col = '#E8E8E8', lwd = 0.8)
  axis(1, at = 1:3, labels = generators, cex.axis = 1.05)
  axis(2, at = seq(0.5, 1.0, 0.1), labels = sprintf('%.1f', seq(0.5, 1.0, 0.1)))
  for (m in seq_along(models)) {
    model <- models[m]
    for (g in seq_along(generators)) {
      rows <- per_gen[per_gen$model == model & per_gen$generator == generators[g], ]
      rows <- rows[match(seeds, rows$seed), ]
      x <- g + (m - 2.5) * 0.20
      avg <- mean(rows$auroc)
      spread <- sd(rows$auroc)
      whisker(x, avg, spread, colors[model])
      points(x + c(-0.025, 0, 0.025), rows$auroc, pch = 16, cex = 0.85,
             col = adjustcolor(colors[model], alpha.f = 0.5))
      points(x, avg, pch = 18, cex = 1.55, col = colors[model])
      text(x, max(rows$auroc, avg + spread) + 0.013, sprintf('%.4f', avg),
           cex = 0.80, col = colors[model])
    }
  }
  title('Detection performance on unseen generators', line = 4.0, cex.main = 1.18)
  legend('top', inset = c(0, -0.16), xpd = NA, legend = model_names[models],
         col = colors[models], pch = 18, ncol = 2, bty = 'n', cex = 0.87)
  mtext('Dots: individual seeds. Diamonds and whiskers: mean +/- SD across 3 seeds.',
        side = 1, line = 3.0, cex = 0.80)
  mtext('Each generator: 2,500 AI images + the same 1,166 real images.',
        side = 1, line = 4.1, cex = 0.80)
})

render('02_mean_and_consistency_B2', 11.0, 5.9, function() {
  par(mfrow = c(1, 2), oma = c(4.2, 0, 2.0, 0), mar = c(3.4, 4.7, 3.5, 1.0))
  for (metric in c('macro_auroc', 'cross_generator_sd')) {
    is_mean <- metric == 'macro_auroc'
    ticks <- if (is_mean) seq(0.5, 1.0, 0.1) else seq(0, 0.12, 0.02)
    plot(NA, xlim = c(0.5, 4.5), ylim = if (is_mean) c(0.5, 1.02) else c(0, 0.12),
         xaxt = 'n', yaxt = 'n', xlab = '',
         ylab = if (is_mean) 'Mean AUROC (higher is better)' else 'Cross-generator SD (lower is better)')
    abline(h = ticks, col = '#E8E8E8', lwd = 0.8)
    axis(1, at = 1:4, labels = models, cex.axis = 1.10)
    axis(2, at = ticks, labels = if (is_mean) sprintf('%.1f', ticks) else sprintf('%.2f', ticks))
    for (m in seq_along(models)) {
      model <- models[m]
      rows <- stability[stability$model == model, ]
      values <- rows[[metric]][match(seeds, rows$seed)]
      avg <- mean(values)
      spread <- sd(values)
      whisker(m, avg, spread, colors[model])
      points(m + c(-0.075, 0, 0.075), values, pch = 16, cex = 0.95,
             col = adjustcolor(colors[model], alpha.f = 0.5))
      points(m, avg, pch = 18, cex = 1.65, col = colors[model])
      text(m, max(values, avg + spread) + if (is_mean) 0.019 else 0.005,
           sprintf('%.4f', avg), col = colors[model], cex = 0.92)
    }
    title(if (is_mean) 'Average detection performance' else 'Cross-generator consistency',
          line = 2.1, cex.main = 1.05)
    mtext(if (is_mean) 'Each seed: mean AUROC across 3 generators' else 'Each seed: generator SD of AUROC (ddof = 0)',
          side = 3, line = 0.7, cex = 0.76)
  }
  mtext('Full test: A, F, B and B2; seeds 42, 123 and 2026', outer = TRUE,
        side = 3, line = 0.6, font = 2, cex = 1.0)
  mtext('Dots: individual seeds. Diamonds and whiskers: mean +/- seed SD (not confidence intervals).',
        outer = TRUE, side = 1, line = 0.8, cex = 0.75)
  mtext('A: spatial only     F: frequency only     B: trainable frequency fusion',
        outer = TRUE, side = 1, line = 2.0, cex = 0.79)
  mtext('B2: frozen frequency fusion', outer = TRUE, side = 1, line = 3.1, cex = 0.79)
})
cat('Created two B2 comparison figures; plotted means and SDs match the score tables.\n')
