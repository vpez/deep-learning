/**
 * This is a linear classifier program, which uses the "Perceptron Learning Algorithm".
 * It works on a 2-D space, thus finding a 1-D hyperplane (line) which classifies the points.
 *
 * Input parameters are:
 * A list of Point2D point objects, containing X and Y coordinates,
 * A list of values, containing 1 or -1 that classify each of the abovementioned points.
 */

import java.awt.geom.*;

int END = -1;

/* Input points */
List<Point2D> points = Arrays.asList(new Point2D.Double(-1, 1), new Point2D.Double(0, -1), new Point2D.Double(10, 1));
List<Double> values = Arrays.asList(1.0, -1.0, 1.0);

/* Initial weights and bias */
Point2D weights = new Point2D.Double(0.0, 0.0);
Double bias = 0.0;

/* Calculates dot-product of vectors and applies bias */
BiFunction<Point2D, Point2D, Double> multiply = (v1, v2) -> v1.getX() * v2.getX() + v1.getY() * v2.getY() + bias;

/* Evaluates the i-th data point against current weights vector and bias value */
Predicate<Integer> evaluate = index -> multiply.apply(points.get(index), weights) * values.get(index) > 0;

/* Updates weights to adapt to the i-th data point */
BiConsumer<Integer, Point2D> updateWeights = (index, w) -> {
    w.setLocation(
            w.getX() + points.get(index).getX() * values.get(index),
            w.getY() + points.get(index).getY() * values.get(index));
};

/* Updates bias to adapt to the i-th data point */
Function<Integer, Double> updateBias = index -> bias + values.get(index);

/* Starting from the i-th index, finds the next index of data points that doesn't evaluate correctly using the current weights and bias values */
Function<Integer, Integer> next = index -> {
        int n = index;
        do {
            n = (n+1) % points.size();
        } while (evaluate.test(n) && n != index);

        return (n == index && evaluate.test(n)) ? END : n;
};

/* Loop over the point set */
for (int i = 0;  i != END && !evaluate.test(i); i = next.apply(i)) {
    updateWeights.accept(i, weights);
    bias = updateBias.apply(i);
}

/* Announce the result */
Function<Point2D, String> weightsToString = p -> "(" + p.getX() + ", " + p.getY() + ")";

System.out.println("weights = " + weightsToString.apply(weights) + ", bias = " + bias);

/exit
