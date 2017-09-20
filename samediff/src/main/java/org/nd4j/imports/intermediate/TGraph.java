package org.nd4j.imports.intermediate;

import lombok.Getter;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This class provides intermediate representation of Graph
 *
 * @author raver119@gmail.com
 */
public class TGraph {
    @Getter protected TVariableSpace variableSpace = new TVariableSpace();

    // this is the layered representation
    protected Map<Integer, List<TNode>> onionMap = new HashMap<>();

    // here we're storing unmapped nodes
    protected List<TNode> unmapped = new ArrayList<>();

    protected void expandOnion(int layer) {
        onionMap.put(layer, new ArrayList<>());
    }

    public void addNode(@NonNull TNode node) {
        unmapped.add(node);
    }
}
