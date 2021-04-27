package cmsc420_s21;

/*This class is gives the primary representation of each of the nodes in the 
AAXTree, including all internal and external nodes. External nodes contain data
about the key, value, level. Internal nodes contain data about the key, leve, and left and right 
children.   */
public class AAXNode<Key extends Comparable<Key>, Value> {

    Key key;
    Value value;
    int level;
    AAXNode<Key, Value> left;
    AAXNode<Key, Value> right;

    /*
     * This constructer is th representation of all of the external nodes, or leaf
     * nodes of the tree. As a result, the constructor utilizes keys and values. I
     * defaulted the level of the external node to 0, as we know that all external
     * nodes fall under the lower most level in the tree.
     */
    public AAXNode(Key key, Value value) {
        this.key = key;
        this.value = value;
        this.left = null;
        this.right = null;
        this.level = 0;
    }

    /*
     * This constructor is the representation of the internal nodes in the AAXTree.
     * I defaulted the value to be null, in order to represent these nodes as being
     * internal nodes with no values. Key value is defaulted, as well as left and
     * right child. Each node starts at a level at 1, and updates when rotations
     * occur.
     */
    public AAXNode(Key key) {
        this.key = key;
        this.value = null;
        this.left = null;
        this.right = null;
        this.level = 1;

    }

}
