package org.nd4j.imports.intermediate;

import org.nd4j.linalg.primitives.ImmutablePair;

/**
 * This class is used as index for TNodes
 *
 * @author raver119@gmail.com
 */
public class TIndex {
    protected ImmutablePair<Integer, Integer> pair;

    protected TIndex() {

    }

    protected TIndex(int node, int index) {
        pair = ImmutablePair.makePair(node, index);
    }

    public static TIndex makeOf(int node, int index) {
        return new TIndex(node, index);
    }

    public static TIndex makeOf(int node) {
        return makeOf(node, 0);
    }

    public int getNode(){
        return pair.getFirst();
    }

    public int getIndex() {
        return pair.getSecond();
    }
}
