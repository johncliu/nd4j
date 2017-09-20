package org.nd4j.imports.intermediate;

import com.google.flatbuffers.FlatBufferBuilder;
import lombok.Getter;
import lombok.NonNull;
import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.nio.ByteBuffer;
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

    protected int getTailSize() {
        return unmapped.size();
    }

    protected void buildOnion() {
        while (getTailSize() > 0) {

        }
    }

    public TGraph provideArrayForVariable(String id, INDArray array) {
        if (!variableSpace.hasVariable(id))
            throw new ND4JIllegalStateException("Unknown variable provided: [" + id + "]");

        variableSpace.getVariable(id).setArray(array);

        return this;
    }

    public ByteBuffer asFlatBuffers() {
        if (variableSpace.hasUndefinedPlaceholders())
            throw new ND4JIllegalStateException("You should provide placeholder values before launching graph");

        FlatBufferBuilder bufferBuilder = new FlatBufferBuilder(0);

        // first of all we build VariableSpace dump


        // then we build onion dump. we don't need it, but why not?
        val keys = onionMap.keySet();
        for (val key: keys) {
            val ops = onionMap.get(key);

            for (val node: ops) {
                // dump right here
            }
        }

        return bufferBuilder.dataBuffer();
    }
}
