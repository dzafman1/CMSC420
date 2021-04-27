package cmsc420_s21; // Don't delete this or your file won't pass the autograder

import java.util.ArrayList;

import javax.management.RuntimeErrorException;

/**
 * AAXTree (skeleton)
 *
 * MODIFY THE FOLLOWING CLASS.
 *
 * You are free to make whatever changes you like or to create additional
 * classes and files, but avoid reusing/modifying the other files given in this
 * folder.
 */

public class AAXTree<Key extends Comparable<Key>, Value> {

	// ------------------------------------------------------------------
	// Needed for Part a
	// ------------------------------------------------------------------
	AAXNode<Key, Value> root;
	int treeSize;

	public AAXTree() {
		AAXNode<Key, Value> root = null; // Empty tree is represented with a root of null.
		treeSize = 0; // Tree size defaulted to 0

	}

	public Value find(Key k) {
		if (root == null) {
			return null;
		}
		return findHelp(k, root);

	}

	public Value findHelp(Key k, AAXNode<Key, Value> currNode) {
		if (currNode.value != null) {
			if (currNode.key.equals(k)) {
				return currNode.value;
			} else {
				return null;
			}
		}
		if (currNode.key.compareTo(k) > 0) {
			return findHelp(k, currNode.left);
		}
		return findHelp(k, currNode.right);
	}

	public void insert(Key x, Value v) throws Exception {
		AAXNode<Key, Value> newNode = new AAXNode<Key, Value>(x, v);
		if (root == null) {
			root = newNode;
			treeSize += 1;
			return;
		}
		if (find(x) != null) {
			throw new Exception("Insertion of duplicate key");

		}
		root = insertHelp(newNode, root);
		treeSize += 1;

	}

	public AAXNode<Key, Value> insertHelp(AAXNode<Key, Value> newNode, AAXNode<Key, Value> currNode) {

		if (currNode.value != null) {
			AAXNode<Key, Value> currNodeData = currNode;

			if (currNode.key.compareTo(newNode.key) > 0) {
				AAXNode<Key, Value> tempNode = new AAXNode<Key, Value>(currNode.key);
				currNode = tempNode;
				currNode.left = newNode;
				currNode.right = currNodeData;
			}
			if (currNode.key.compareTo(newNode.key) < 0) {
				AAXNode<Key, Value> tempNode = new AAXNode<Key, Value>(newNode.key);
				currNode = tempNode;
				currNode.left = currNodeData;
				currNode.right = newNode;

			}
		} else {

			if (currNode.key.compareTo(newNode.key) > 0) {
				currNode.left = insertHelp(newNode, currNode.left);
			} else {
				currNode.right = insertHelp(newNode, currNode.right);
			}

		}
		return split(skew(currNode));

	}

	public AAXNode<Key, Value> split(AAXNode<Key, Value> root) {
		if (root.right == null || root.right.right == null) {
			return root;
		}

		if (root.right.right.level == root.level && root.right.level == root.level) {
			AAXNode<Key, Value> q = root.right;
			root.right = q.left;
			q.left = root;
			q.level += 1;
			return q;

		}

		return root;

	}

	public AAXNode<Key, Value> skew(AAXNode<Key, Value> root) {

		if (root.left == null) {
			return root;
		}

		if (root.left.level == root.level) {
			AAXNode<Key, Value> q = root.left;
			root.left = q.right;
			q.right = root;
			return q;
		}
		return root;

	}

	public void clear() {
		root = null;
		treeSize = 0;
	}

	public ArrayList<String> getPreorderList() {
		if (root == null) {
			return new ArrayList<String>();
		}
		return preorderHelp(root, new ArrayList<String>());

	}

	public ArrayList<String> preorderHelp(AAXNode<Key, Value> currNode, ArrayList<String> ret) {

		if (currNode == null) {
			return ret;
		}
		if (currNode.value != null) {
			ret.add("[" + currNode.key + " " + currNode.value + "]");
			return ret;
		}
		ret.add("(" + currNode.key + ") " + currNode.level);
		ret = preorderHelp(currNode.left, ret);
		ret = preorderHelp(currNode.right, ret);
		return ret;
	}

	// ------------------------------------------------------------------
	// Needed for Part b
	// ------------------------------------------------------------------

	/*
	 * If key doesn't exist in the tree, exception is thrown. Otherwise, call
	 * deleteHelp and set root of tree to be the return of the function. With
	 * successful deletion, treeSize is decremented.
	 */
	public void delete(Key x) throws Exception {
		if (find(x) == null) {
			throw new Exception("Deletion of nonexistent key");
		}
		root = deleteHelp(x, root);
		treeSize -= 1;
	}

	public AAXNode<Key, Value> deleteHelp(Key x, AAXNode<Key, Value> currNode) {
		/* If key is less than currNode, check left child. */
		if (currNode.key.compareTo(x) > 0 && currNode.value == null) {
			currNode.left = deleteHelp(x, currNode.left);
			/*
			 * If reached external node from deleteHelp recursive call, return right child.
			 */
			if (currNode.left == null) {
				return currNode.right;
			}
		}
		/* If key is less than currNode, check right child. */
		if (currNode.key.compareTo(x) <= 0 && currNode.value == null) {
			currNode.right = deleteHelp(x, currNode.right);
			/*
			 * If reached external node from deleteHelp recursive call, return left child.
			 */
			if (currNode.right == null) {
				return currNode.left;
			}
		} else {
			/* If external node reached, return null */
			if (currNode.value != null) {
				return null;
			}
		}
		/* Fix tree after deletion occurred */
		return fixAfterDelete(currNode);

	}

	public AAXNode<Key, Value> fixAfterDelete(AAXNode<Key, Value> p) {
		updateLevel(p);
		p = skew(p);
		p.right = skew(p.right);
		if (p.right.right != null) {
			p.right.right = skew(p.right.right);
		}
		p = split(p);
		p.right = split(p.right);
		return p;
	}

	public void updateLevel(AAXNode<Key, Value> p) {
		int idealLevel = 1 + Math.min(p.left.level, p.right.level);
		if (p.level > idealLevel) {
			p.level = idealLevel;
			if (p.right.level > idealLevel) {
				p.right.level = idealLevel;
			}
		}
	}

	/* Returns size of tree */
	public int size() {
		return treeSize;
	}

	/* If empty tree, return null. Otherwise, return minHelp */
	public Value getMin() {
		if (root == null) {
			return null;
		}
		return minHelp(root);
	}

	/* Search for left most child of left subtree. Return external node value. */
	public Value minHelp(AAXNode<Key, Value> currNode) {
		if (currNode.left == null) {
			return currNode.value;
		}
		return minHelp(currNode.left);
	}

	/* If empty tree, return null. Otherwise, return maxHelp */
	public Value getMax() {
		if (root == null) {
			return null;
		}
		return maxHelp(root);
	}

	/* Search for right most child of right subtree. Return external node value. */
	public Value maxHelp(AAXNode<Key, Value> currNode) {
		if (currNode.right == null) {
			return currNode.value;
		}
		return maxHelp(currNode.right);
	}

	/* If empty tree, return null. Otherwise, return findSmaller */
	public Value findSmaller(Key x) {
		if (root == null) {
			return null;
		}
		return findSmallerHelper(x, root);
	}

	public Value findSmallerHelper(Key x, AAXNode<Key, Value> currNode) {
		/*
		 * If reached external node, and currNode key is less than x, return currNode
		 * value. Else, return null.
		 */
		if (currNode.value != null) {
			if (currNode.key.compareTo(x) < 0) {
				return currNode.value;
			}
			return null;
		}
		/* If currNode key less than or equal to x, traverse right subtree. */
		if (currNode.key.compareTo(x) <= 0) {
			Value temp = findSmallerHelper(x, currNode.right);
			/* If temp value null, perform secondarySearch */
			if (temp == null) {
				return secondarySearchSmaller(currNode.left);
			}
			return temp;
			/* If currNode key greater than x, traverse left subtree. */
		} else if (currNode.key.compareTo(x) > 0) {
			return findSmallerHelper(x, currNode.left);
		}
		return null;
	}

	/* Return right most value from right subtree. */
	public Value secondarySearchSmaller(AAXNode<Key, Value> currNode) {
		if (currNode.value != null) {
			return currNode.value;
		}
		return secondarySearchSmaller(currNode.right);
	}

	/* If empty tree, return null. Otherwise, return findLarger */
	public Value findLarger(Key x) {
		if (root == null) {
			return null;
		}
		return findLargerHelper(x, root);
	}

	public Value findLargerHelper(Key x, AAXNode<Key, Value> currNode) {
		/*
		 * If reached external node, and currNode key is greater than x, return currNode
		 * value. Else, return null.
		 */
		if (currNode.value != null) {
			if (currNode.key.compareTo(x) > 0) {
				return currNode.value;
			}
			return null;
		}
		/* If currNode key less than x, traverse left subtree. */
		if (currNode.key.compareTo(x) <= 0) {
			return findLargerHelper(x, currNode.right);

			/* If currNode key greater than or equal to x, traverse left subtree. */
		} else if (currNode.key.compareTo(x) > 0) {
			Value temp = findLargerHelper(x, currNode.left);

			/* If temp value null, perform secondarySearch */
			if (temp == null) {
				return secondarySearchLarger(currNode.right);
			}
			return temp;
		}
		return null;
	}

	/* Return left most value from left subtree. */
	public Value secondarySearchLarger(AAXNode<Key, Value> currNode) {
		if (currNode.value != null) {
			return currNode.value;
		}
		return secondarySearchLarger(currNode.left);
	}

	/*
	 * If empty tree, return null. Else, find left most child of left subtree.
	 * Perform delete on the key and return value from that node.
	 */
	public Value removeMin() {
		if (root == null) {
			return null;
		}
		AAXNode<Key, Value> retVal = removeMinHelp(root);
		try {
			delete(retVal.key);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

		return retVal.value;

	}

	/* Find left most node of left subtree. Return the node. */
	public AAXNode<Key, Value> removeMinHelp(AAXNode<Key, Value> currNode) {
		if (currNode.value != null) {
			return currNode;
		}
		return removeMinHelp(currNode.left);
	}

	/*
	 * If empty tree, return null. Else, find right most child of right subtree.
	 * Perform delete on the key and return value from that node.
	 */
	public Value removeMax() {
		if (root == null) {
			return null;
		}
		AAXNode<Key, Value> retVal = removeMaxHelp(root);
		try {
			delete(retVal.key);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

		return retVal.value;

	}

	/* Find right most node of right subtree. Return the node. */
	public AAXNode<Key, Value> removeMaxHelp(AAXNode<Key, Value> currNode) {
		if (currNode.right == null) {
			return currNode;
		}
		return removeMaxHelp(currNode.right);
	}

}
