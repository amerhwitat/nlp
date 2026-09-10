package org.chimera.thamudic;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
class ThamudicTest{@Test void detects(){assertTrue(Thamudic.isThamudic(0x10A80));assertFalse(Thamudic.isThamudic('A'));assertEquals("𐪀𐪁",Thamudic.extract("A𐪀B𐪁"));}}
