#' convert spatial connectivity matrix to edge list (graph structure)
#'
#' @param w connectivity matrix (n x n)
#'
#' @author Connor Donegan
edges <- function (w, both = FALSE) {
    lw <- apply(w, 1, function(r) {
        which(r != 0)
    })
    all.edges <- lapply(1:length(lw), function(i) {
        nbs <- lw[[i]]
        if (length(nbs)) 
            data.frame(node1 = i, node2 = nbs, weight = w[i, 
                nbs])
    })
    edges <- do.call("rbind", all.edges)
    if (!both) edges <- edges[which(edges$node1 < edges$node2), ]
    return(edges)
}

#' connect observations, identified by GEOID
#' @param id a GEOID to identify the county to connect to its `neighbors'
#' @param neighbors character vector of neighboring county GEOIDs
#' @param C connectivity matrix
#' @param data  the data, with columns GEOID and county (name)
#'
#' @author Connor Donegan
connect <- function(id, neighbors, C, data) {
    stopifnot(all(dim(C) == nrow(data)))
    id1 <- which(data$GEOID == id)
    for (i in seq_along(neighbors)) {
        id2 <- which(data$GEOID == neighbors[i])
        message("Connecting ", as.data.frame(data)[c(id1, id2), c("GEOID", "county")],"\n")
        C[id1, id2] <- C[id2, id1] <- 1
    }
    return(C)
}

#' Prepare data for Stan CAR model
#'
#' @param A  Binary adjacency matrix
#' @param lambda  If TRUE, return eigenvalues required for calculating the log determinant
#' of the precision matrix and for determining the range of permissible values of rho.
#' @param cmat  Return the full matrix C if TRUE.
#' 
#' @details
#'
#' The CAR model is Gauss(Mu, Sigma), Sigma = (I - rho C)^{-1} M.
#' This function implements the specification of C and M known as the
#' "neighborhoods" or "weighted" (WCAR) specification (see Cressie and Wikle 2011,
#' pp. 186-88, for CAR specifications).
#'
#' @source
#'  Cressie, N. and C. K. Wikle. Statistics for Spatio-Temporal Data. Wiley.
#'
#' @return A list containing all of the data elements required by the Stan CAR model.
#'
#' @author Connor Donegan (Connor.Donegan@UTDallas.edu)
#' 
prep_car_data <- function(A, lambda = FALSE, cmat = TRUE) {
    n <- nrow(A)    
    Ni <- rowSums(A)
    C <- A / Ni
    M_diag <- 1 / Ni
    stopifnot( isSymmetric.matrix(C %*% diag(M_diag), check.attributes = FALSE) )
    car.dl <- rstan::extract_sparse_parts(diag(n) - C)
    car.dl$Cidx <- which( car.dl$w != 1 )
    car.dl$nImC <- length(car.dl$w)
    car.dl$nC <- length(car.dl$Cidx)
    car.dl$M_diag <- M_diag
    car.dl$n <- n
    if (lambda) {
        MCM <- diag( 1 / sqrt(M_diag) ) %*% C %*% diag( sqrt(M_diag) )
        lambda <- eigen(MCM)$values
        cat ("Range of permissible rho values: ", 1 / range(lambda), "\n")
        car.dl$lambda <- lambda
    }
    if (cmat) car.dl$C <- C
    return (car.dl)
}
#' Caclulate standard error for log(x)
#'
#' @param x vector of data values
#' @param se standard errors of x
#' @param method monte carlo method ("mc") or the delta method ("delta")
#' @param nsim number of iterations for the Monte Carlo method
#' @param bounds only positive values will be simulated; set custom boundaries with this argument, such as \code{bounds = c(0, 100)} for percentages.
#'
#' @return Standard errors for the log of a variable, based on standard errors of the original variable.
#'
#' @details If the truncnorm package is not available for some reason, results may be approximately reproduced by changing the method from "mc" to "delta".
#'
#' @author Connor Donegan
se_log <- function (x, se, method = c("mc", "delta"), nsim = 30000, bounds = c(0, 
    Inf)) {
    stopifnot(length(x) == length(se))
    method <- match.arg(method)
    if (method == "mc") {
        stopifnot(bounds[1] >= 0 & bounds[1] < bounds[2])
        se.log <- NULL
        for (i in seq_along(x)) {
            z <- truncnorm::rtruncnorm(n = nsim, mean = x[i], 
                sd = se[i], a = bounds[1], b = bounds[2])
            l.z <- log(z)
            se.log <- c(se.log, sd(l.z))
        }
        return(se.log)
    }
    if (method == "delta") 
        return(x^(-1) * se)
}

#' Data model diagnostics
#' 
#' @param fit fitted Rstan observational error model
#' @param z raw data values
#' @param sf simple features (sf) object for mapping z
#' @param W spatial weights matrix
#' @param I Moran's I values
#' 
#' @author Connor Donegan
plot.res <- function(fit, z, sf, W, plot = TRUE, index = FALSE) {
    delta <- as.matrix(fit, pars = "delta")
    delta.mu <- apply(delta, 2, mean)    
    x <- as.matrix(fit, pars = "x")
    x.mu <- apply(x, 2, mean)
    x.lwr <- apply(x, 2, function(c) quantile(c, probs = 0.025))
    x.upr <- apply(x, 2, function(c) quantile(c, probs = 0.975))

    base_size = 13
    theme_cust <- theme_classic(
        base_size = base_size
    )
    
    d.mu <- delta.mu ### as.numeric(scale(delta.mu))
    w.d.mu <- as.numeric(W %*% delta.mu)
        g.mc <- moran_plot(delta.mu, W,
                           size = 1,
                           xlab =  expression(paste(Delta, ' (centered)')),
                           ylab = "Spatial Lag") +
            theme_cust
    g.pint <- data.frame(
        z = z,
        xmu = x.mu,
        xlwr = x.lwr,
        xupr = x.upr,
        id = 1:length(z)
    ) %>% 
        ggplot() +
        geom_pointrange(
        aes(z, y = xmu, ymin = xlwr, ymax = xupr),
        lwd = 0.1,
        position = "jitter"
    ) +
        geom_abline(slope = 1, intercept = 0, lty = 2, col = 'red') +
#        scale_x_continuous(labels = signs::signs_format()) +
#        scale_y_continuous(labels = signs::signs_format()) +        
        theme_cust +
        labs(x = "ACS Estimate", y = "Posterior Mean, 95% C.I.")
    if (index) g.pint = g.pint + geom_label(aes(x=z,y=xmu,label  = id))
        
    d0 <- delta %>%
        as.data.frame %>%
        pivot_longer(
            everything(),
            values_to = "value",
            names_to = "name"
        ) %>%
        group_by(name) %>%
        summarise(
            delta = mean(value),
            lwr = quantile(value, probs = 0.025),
            upr = quantile(value, probs = 0.975)
            ) %>%
        mutate(
            idx = as.integer(str_extract(name, "[:digit:]+"))
        )
    df$idx <- 1:nrow(df)
    d0 <- dplyr::inner_join(df, d0, by = "idx")
        g.map <- d0 %>%
            ggplot() +
            geom_sf(
                aes(fill = delta),
                lwd = 0.05
        ) +
            scale_fill_gradient2(
                name = expression(Delta)#,
#              labels = signs::signs_format()
            ) +
            theme_void() +
            theme(
                legend.position = "right",
                legend.key.size = unit(1,"line"),
                legend.text = element_text(size = 0.9 * base_size)
                )
    if (plot) gridExtra::grid.arrange(g.pint, g.mc, g.map, ncol = 1)
    if (!plot) gridExtra::arrangeGrob(g.pint, g.mc, g.map, ncol = 1)
}

#' Moran plot for spatial autocorrelation
#' @param y variable of interest
#' @param w spatial weights matrix
#' @param xlab x-axis title/label
#' @param ylab y-axis title/label
#'
#' @return a ggplot with y - mean(y) on the x-axis and its spatially lagged value on the y-axis (w %*% y).
#'
#' @author Connor Donegan
moran_plot <- function (y, w, xlab = "y (centered)", ylab = "Spatial Lag", 
    pch = 20, col = "darkred", size = 2, alpha = 1, lwd = 0.5) {
    if (!(inherits(y, "numeric") | inherits(y, "integer"))) 
        stop("y must be a numeric or integer vector")
    sqr <- all(dim(w) == length(y))
    if (!inherits(w, "matrix") | !sqr) 
        stop("w must be an n x n matrix where n = length(y)")
    if (any(rowSums(w) == 0)) {
        zero.idx <- which(rowSums(w) == 0)
        message(length(zero.idx), " observations with no neighbors found. They will be dropped from the data.")
        y <- y[-zero.idx]
        w <- w[-zero.idx, -zero.idx]
    }
    y <- y - mean(y)
    ylag <- as.numeric(w %*% y)
    sub <- paste0("MC = ", round(mc(y, w), 3))
    ggplot(data.frame(y = y, ylag = ylag),
           aes(x = y, y = ylag)
           ) + 
        geom_hline(yintercept = mean(ylag), lty = 3) +
        geom_vline(xintercept = mean(y), 
                   lty = 3) +
        geom_point(pch = 20, colour = col, size = size, 
                                         alpha = alpha, aes(x = y, y = ylag)) +
        geom_smooth(aes(x = y, 
                        y = ylag),
                    method = "lm",
                    lwd = lwd,
                    col = "black",
                    se = FALSE) + 
        labs(x = xlab, y = ylab, subtitle = sub) +
 #       scale_x_continuous(labels = signs::signs_format()) +
#        scale_y_continuous(labels = signs::signs_format()) +
        theme_classic()
}
    
#' Moran coefficient
#'
#'
#' @author Connor Donegan
mc <- function (x, w, digits = 3, warn = TRUE) {
    if (missing(x) | missing(w)) 
        stop("Must provide data x (length n vector) and n x n spatial weights matrix (w).")
    if (any(rowSums(w) == 0)) {
        zero.idx <- which(rowSums(w) == 0)
        if (warn) 
            message(length(zero.idx), " observations with no neighbors found. They will be dropped from the data.")
        x <- x[-zero.idx]
        w <- w[-zero.idx, -zero.idx]
    }
    xbar <- mean(x)
    z <- x - xbar
    ztilde <- as.numeric(w %*% z)
    A <- sum(rowSums(w))
    n <- length(x)
    mc <- as.numeric(n/A * (z %*% ztilde)/(z %*% z))
    return(round(mc, digits = digits))
}


#' prepare Stan data for ICAR model given a connectivity matrix
#' 
#' @param C a connectivity matrix
#' @param scale_factor optional vector of scale factors for each connected portion of the graph structure. 
#'   Generally, you will ignore this and update the scale factor manually.
#'   
#' @return a list with all that is needed for the Stan ICAR prior. If you do not provide inv_sqrt_scale_factor, 
#'   it will be set to a vector of 1s.
#'   
#' @author Connor Donegan
#' 
prep_icar_data <- function (C, inv_sqrt_scale_factor = NULL) {
  n <- nrow(C)
  E <- edges(C)
  G <- list(np = nrow(C), from = E$node1, to = E$node2, nedges = nrow(E))
  class(G) <- "Graph"
  nb2 <- spdep::n.comp.nb(spdep::graph2nb(G))
  k = nb2$nc
  if (inherits(inv_sqrt_scale_factor, "NULL")) inv_sqrt_scale_factor <- array(rep(1, k), dim = k)
  group_idx = NULL
  for (j in 1:k) group_idx <- c(group_idx, which(nb2$comp.id == j))
  group_size <- NULL
  for (j in 1:k) group_size <- c(group_size, sum(nb2$comp.id == j))
  # intercept per connected component of size > 1, if multiple.
  m <- sum(group_size > 1) - 1
  if (m) {
    GS <- group_size
    ID <- nb2$comp.id
    change.to.one <- which(GS == 1)
    ID[which(ID == change.to.one)] <- 1
    A = model.matrix(~ factor(ID))
    A <- as.matrix(A[,-1])
  } else {
    A <- model.matrix(~ 0, data.frame(C))
  }
  l <- list(k = k, 
            group_size = array(group_size, dim = k), 
            n_edges = nrow(E), 
            node1 = E$node1, 
            node2 = E$node2, 
            group_idx = array(group_idx, dim = n), 
            m = m,
            A = A,
            inv_sqrt_scale_factor = inv_sqrt_scale_factor, 
            comp_id = nb2$comp.id)
  return(l)
}

#' compute scaling factor for adjacency matrix
#' accounts for differences in spatial connectivity 
#' 
#' @param C connectivity matrix
#' 
#' Requires the following packages: 
#' 
#' library(Matrix)
#' library(INLA);
#' library(spdep)
#' library(igraph)
#' 
#' @author Mitzi Morris
#' 
scale_c <- function(C) {
  #' compute geometric mean of a vector
  geometric_mean <- function(x) exp(mean(log(x))) 
  
  N = dim(C)[1]
  
  # Create ICAR precision matrix  (diag - C): this is singular
  # function Diagonal creates a square matrix with given diagonal
  Q =  Diagonal(N, rowSums(C)) - C
  
  # Add a small jitter to the diagonal for numerical stability (optional but recommended)
  Q_pert = Q + Diagonal(N) * max(diag(Q)) * sqrt(.Machine$double.eps)
  
  # Function inla.qinv provides efficient way to calculate the elements of the
  # the inverse corresponding to the non-zero elements of Q
  Q_inv = inla.qinv(Q_pert, constr=list(A = matrix(1,1,N),e=0))
  
  # Compute the geometric mean of the variances, which are on the diagonal of Q.inv
  scaling_factor <- geometric_mean(Matrix::diag(Q_inv)) 
  return(scaling_factor) 
}

#' Download and unzip a shapefile into your working directory
#' 
#' @param url URL to a shapefile. If you're downloading a shapefil manually, right click the "Download" button and copy the URL to your clipboard.
#' @param folder name of the folder to unzip the file into (character string).
#' 
#' @return a shapefile (or whatever you provided a URL to) in your working directory; also prints the contents of the folder with full file paths.
#' 
#' @author Connor Donegan
#' 
get_shp <- function(url, folder = "shape") {
  tmp.dir <- tempfile(fileext = ".zip")
  download.file(url, destfile = tmp.dir)
  unzip(tmp.dir, exdir = folder)
  list.files(folder, full.names = TRUE)
}

