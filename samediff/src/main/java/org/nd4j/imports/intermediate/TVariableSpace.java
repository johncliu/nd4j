package org.nd4j.imports.intermediate;

import lombok.NonNull;
import lombok.val;
import org.nd4j.linalg.primitives.ImmutablePair;
import org.nd4j.linalg.primitives.Pair;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This class is intermediate representation of VariableSpace for Graph
 *
 * @author raver119@gmail.com
 */
public class TVariableSpace {
    protected Map<TIndex, TVariable> numericMap = new HashMap<>();
    protected Map<String, TVariable> symbolicMap = new HashMap<>();

    protected List<TVariable> placeholders;


    public void addVariable(int id, @NonNull TVariable variable) {
        val key = TIndex.makeOf(id, 0);
        numericMap.put(key, variable);

        if (variable.getName() != null && !variable.getName().isEmpty())
            symbolicMap.put(variable.getName(), variable);

        if (variable.isPlaceholder())
            placeholders.add(variable);
    }

    public void addVariable(@NonNull String id, @NonNull TVariable variable) {
        symbolicMap.put(id, variable);
    }


    public boolean hasVariable(int id) {
        return hasVariable(TIndex.makeOf(id, 0));
    }

    public boolean hasVariable(@NonNull TIndex id) {
        return numericMap.containsKey(id);
    }

    public boolean hasVariable(@NonNull String id) {
        return symbolicMap.containsKey(id);
    }


    public TVariable getVariable(int id) {
        return getVariable(TIndex.makeOf(id, 0));
    }

    public TVariable getVariable(@NonNull TIndex id) {
        return numericMap.get(id);
    }

    public TVariable getVariable(String id) {
        return symbolicMap.get(id);
    }

    public void clear() {
        numericMap.clear();
        symbolicMap.clear();;
        placeholders.clear();
    }
}
